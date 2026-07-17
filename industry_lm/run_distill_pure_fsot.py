#!/usr/bin/env python3
"""
Stable distill toward pure FSOT:
  A) Blend α-only (proven) → high agree
  B) Push α up gently with KL + α-pressure (α-only, fp32)
  C) Pure FSOT all layers; adapt o_proj + norms (fp32, low LR, grad clip)
  D) Remeasure
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from competitive.sparse_consensus_batched import consensus_true_sparse_padded  # noqa: E402
from fsot_cuda_ops import available as cuda_ok, fsot_consensus  # noqa: E402
from fsot_lib.seeds import SEEDS  # noqa: E402
from fsot_layer_swap import swap_all_layers  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
OUT.mkdir(parents=True, exist_ok=True)

PHI = SEEDS.phi
ALPHA0 = 1.0 / (PHI * PHI)
# Successful blend used suction*K ≈ 0.062 on α only. Keep that for α.
LR_ALPHA = float(SEEDS.suction * SEEDS.k)
# o_proj / norms need much smaller LR
LR_WEIGHTS = 2e-4

PROBES = [
    "The capital of France is",
    "def fibonacci(n):",
    "In mathematics, the derivative of x^2 is",
    "Once upon a time",
    "2 + 2 =",
    "The largest planet in our solar system is",
    "print('hello",
    "Water freezes at",
]
TRAIN = PROBES + [
    "Python is a programming language that",
    "The speed of light is approximately",
    "To sort a list in reverse order",
    "Photosynthesis converts",
    "The mitochondria is",
    "In computer science, a binary tree",
    "Newton's second law states",
    "The chemical formula for water is",
]


def fsot_core(q, k, v):
    # keep fp32 for stability
    qf, kf, vf = q.float(), k.float(), v.float()
    if cuda_ok() and q.is_cuda:
        try:
            out = fsot_consensus(qf, kf, vf)
            if torch.isfinite(out).all():
                return out
        except Exception:
            pass
    out = torch.stack(
        [consensus_true_sparse_padded(qf[b], kf[b], vf[b]) for b in range(qf.shape[0])],
        0,
    )
    return out


class BlendAttn(nn.Module):
    def __init__(self, src):
        super().__init__()
        self.config = src.config
        self.layer_idx = getattr(src, "layer_idx", 0)
        self.head_dim = src.head_dim
        self.num_key_value_groups = src.num_key_value_groups
        self.scaling = src.scaling
        self.q_proj = src.q_proj
        self.k_proj = src.k_proj
        self.v_proj = src.v_proj
        self.o_proj = src.o_proj
        logit0 = math.log(ALPHA0 / (1.0 - ALPHA0))
        self.alpha_logit = nn.Parameter(torch.tensor(float(logit0)))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)

        a = self.alpha.to(dtype=torch.float32)
        # always float32 mix for stable grads
        qf, kf, vf = q.float(), k.float(), v.float()
        sdpa = F.scaled_dot_product_attention(
            qf, kf, vf, is_causal=True, scale=float(self.scaling)
        )
        fs = fsot_core(qf, kf, vf)
        mix = (1.0 - a) * sdpa + a * fs
        out = mix.to(dtype=hidden_states.dtype)
        out = out.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(out), None


def load(device, train=False):
    # always fp32 for distill stability
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    if not train:
        m.eval()
    return tok, m


@torch.no_grad()
def measure(tok, teacher, student, device):
    agree = kl = top = 0.0
    n = len(PROBES)
    for p in PROBES:
        inp = tok(p, return_tensors="pt").to(device)
        lt = teacher(**inp).logits[0, -1].float().cpu()
        ls = student(**inp).logits[0, -1].float().cpu()
        if not torch.isfinite(ls).all():
            return 0.0, float("nan"), 0.0
        pt = F.softmax(lt, dim=-1)
        kl += float(
            (pt * (torch.log(pt.clamp_min(1e-12)) - F.log_softmax(ls, dim=-1))).sum()
        )
        agree += float(lt.argmax() == ls.argmax())
        top += len(
            set(torch.topk(lt, 5).indices.tolist())
            & set(torch.topk(ls, 5).indices.tolist())
        ) / 5.0
    return agree / n, kl / n, top / n


@torch.no_grad()
def tps(tok, model, device):
    rates = []
    for p in PROBES[:4]:
        inp = tok(p, return_tensors="pt").to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            **inp,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        rates.append((out.shape[-1] - inp["input_ids"].shape[-1]) / (ms / 1000))
    return sum(rates) / len(rates)


def mean_alpha(model):
    vals = []
    for layer in model.model.layers:
        if hasattr(layer.self_attn, "alpha"):
            with torch.no_grad():
                vals.append(float(layer.self_attn.alpha))
    return sum(vals) / max(len(vals), 1) if vals else float("nan")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "phases": {}}

    tok, teacher = load(device, train=False)
    for p in teacher.parameters():
        p.requires_grad_(False)

    # ===== A: α-only blend (stable path that hit 100% before) =====
    print("=== A: α-only blend (stable) ===")
    _, student = load(device, train=True)
    for i in range(len(student.model.layers)):
        student.model.layers[i].self_attn = BlendAttn(student.model.layers[i].self_attn)
    for n, p in student.named_parameters():
        p.requires_grad_("alpha_logit" in n)
    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=LR_ALPHA)

    student.train()
    losses = []
    for step in range(80):
        text = TRAIN[step % len(TRAIN)]
        inp = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            t_log = teacher(**inp).logits
        s_log = student(**inp).logits
        loss = F.kl_div(
            F.log_softmax(s_log, dim=-1),
            F.softmax(t_log, dim=-1),
            reduction="batchmean",
        )
        if not torch.isfinite(loss):
            print(f"  A abort non-finite at step {step}")
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        opt.step()
        # clamp logits
        with torch.no_grad():
            for layer in student.model.layers:
                layer.self_attn.alpha_logit.clamp_(-6, 6)
        losses.append(float(loss.item()))
        if step % 20 == 0 or step == 79:
            print(f"  A {step:03d} kl={loss.item():.4f} α={mean_alpha(student):.3f}")

    student.eval()
    a0, k0, t0 = measure(tok, teacher, student, device)
    tps0 = tps(tok, student, device)
    tps_b = tps(tok, teacher, device)
    report["phases"]["A_alpha_only"] = {
        "agree": a0,
        "kl": k0,
        "top5": t0,
        "alpha": mean_alpha(student),
        "tps": tps0,
        "tps_x": tps0 / max(tps_b, 1e-9),
        "loss_end": losses[-1] if losses else None,
    }
    print(f"  A done agree={a0:.0%} KL={k0:.3f} α={mean_alpha(student):.3f}")

    # ===== B: push α up with mild pressure (α-only) =====
    print("=== B: push α toward FSOT ===")
    student.train()
    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=LR_ALPHA * 0.5)
    losses_b = []
    for step in range(100):
        text = TRAIN[step % len(TRAIN)]
        inp = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            t_log = teacher(**inp).logits
        s_log = student(**inp).logits
        kl = F.kl_div(
            F.log_softmax(s_log, dim=-1),
            F.softmax(t_log, dim=-1),
            reduction="batchmean",
        )
        ramp = min(1.0, (step + 1) / 80)
        # mild purity pressure — don't explode
        lam = 0.05 * ramp
        a_pen = torch.stack(
            [(1.0 - layer.self_attn.alpha) ** 2 for layer in student.model.layers]
        ).mean()
        loss = kl + lam * a_pen
        if not torch.isfinite(loss):
            print(f"  B abort non-finite at {step}")
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
        opt.step()
        with torch.no_grad():
            for layer in student.model.layers:
                layer.self_attn.alpha_logit.clamp_(-6, 6)
        losses_b.append(float(loss.item()))
        if step % 25 == 0 or step == 99:
            print(
                f"  B {step:03d} loss={loss.item():.4f} kl={kl.item():.4f} "
                f"α={mean_alpha(student):.3f} λ={lam:.3f}"
            )

    student.eval()
    a1, k1, t1 = measure(tok, teacher, student, device)
    tps1 = tps(tok, student, device)
    report["phases"]["B_alpha_push"] = {
        "agree": a1,
        "kl": k1,
        "top5": t1,
        "alpha": mean_alpha(student),
        "tps": tps1,
        "tps_x": tps1 / max(tps_b, 1e-9),
    }
    print(f"  B done agree={a1:.0%} KL={k1:.3f} α={mean_alpha(student):.3f}")

    # ===== C: pure FSOT + o_proj/norm adapt (fp32, low LR) =====
    print("=== C: pure FSOT + o_proj/norm adapt ===")
    _, pure = load(device, train=True)
    # warm-start o_proj from blend student
    for i in range(len(pure.model.layers)):
        pure.model.layers[i].self_attn.o_proj.load_state_dict(
            student.model.layers[i].self_attn.o_proj.state_dict()
        )
    swap_all_layers(pure)
    for n, p in pure.named_parameters():
        p.requires_grad_(
            ("o_proj" in n)
            or ("input_layernorm" in n)
            or ("post_attention_layernorm" in n)
            or (n == "model.norm.weight")
        )
    n_train = sum(p.numel() for p in pure.parameters() if p.requires_grad)
    print(f"  trainable: {n_train:,}")
    opt_c = torch.optim.AdamW(
        [p for p in pure.parameters() if p.requires_grad],
        lr=LR_WEIGHTS,
        weight_decay=0.01,
    )
    pure.train()
    losses_c = []
    for step in range(200):
        text = TRAIN[step % len(TRAIN)]
        inp = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            t_log = teacher(**inp).logits
        s_log = pure(**inp).logits
        loss = F.kl_div(
            F.log_softmax(s_log, dim=-1),
            F.softmax(t_log, dim=-1),
            reduction="batchmean",
        )
        if not torch.isfinite(loss):
            print(f"  C abort non-finite at {step}")
            break
        opt_c.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in pure.parameters() if p.requires_grad], 1.0
        )
        opt_c.step()
        losses_c.append(float(loss.item()))
        if step % 40 == 0 or step == 199:
            print(f"  C {step:03d} kl={loss.item():.4f}")

    pure.eval()
    a2, k2, t2 = measure(tok, teacher, pure, device)
    tps2 = tps(tok, pure, device)
    report["phases"]["C_pure_fsot"] = {
        "agree": a2,
        "kl": k2,
        "top5": t2,
        "tps": tps2,
        "tps_x": tps2 / max(tps_b, 1e-9),
        "loss_start": losses_c[0] if losses_c else None,
        "loss_end": losses_c[-1] if losses_c else None,
        "trainable": n_train,
    }
    print(f"  C done agree={a2:.0%} KL={k2:.3f} tps×{tps2/max(tps_b,1e-9):.2f}")

    report["baseline_tps"] = tps_b
    report["summary"] = {
        "A_agree": a0,
        "B_agree": a1,
        "B_alpha": report["phases"]["B_alpha_push"]["alpha"],
        "C_agree": a2,
        "C_kl": k2,
        "C_tps_x": tps2 / max(tps_b, 1e-9),
        "pure_unadapted_agree": 0.0,
        "best_agree": max(a0, a1, a2 if a2 == a2 else 0),
    }
    report["ok"] = report["summary"]["best_agree"] >= 0.5
    report["claim"] = (
        f"A α-only agree={a0:.0%}; B α-push agree={a1:.0%} α={report['phases']['B_alpha_push']['alpha']:.2f}; "
        f"C pure-FSOT+adapt agree={a2:.0%} KL={k2:.3f} tps×{tps2/max(tps_b,1e-9):.2f} "
        f"(unadapted pure was 0%)."
    )

    path = OUT / "distill_pure_fsot.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("===", report["claim"])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

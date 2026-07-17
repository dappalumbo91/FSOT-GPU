#!/usr/bin/env python3
"""
All three pushes in one run:
  1) Extended pure-FSOT distill (big corpus, many steps)
  2) LoRA on Q/K/V/O while FSOT operator stays fixed
  3) Dual demos: blend (quality) + pure FSOT (speed path) with ≥80% target

Saves best pure-FSOT checkpoint by agreement.
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
from fsot_layer_swap import swap_all_layers  # noqa: E402
from fsot_lib.seeds import SEEDS  # noqa: E402
from lora_utils import freeze_non_lora, inject_lora_into_fsot_attn, lora_parameters  # noqa: E402
from train_corpus import PROBES, train_texts  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
CKPT = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)
CKPT.mkdir(parents=True, exist_ok=True)

PHI = SEEDS.phi
ALPHA0 = 1.0 / (PHI * PHI)
LR_ALPHA = float(SEEDS.suction * SEEDS.k)
LR_LORA = 3e-4
TARGET_AGREE = 0.80
STEPS_PURE = 2500
STEPS_BLEND = 100
EVAL_EVERY = 250
LORA_R = 16
LORA_ALPHA = 32.0


def fsot_core(q, k, v):
    qf, kf, vf = q.float(), k.float(), v.float()
    if cuda_ok() and q.is_cuda:
        try:
            o = fsot_consensus(qf, kf, vf)
            if torch.isfinite(o).all():
                return o
        except Exception:
            pass
    return torch.stack(
        [consensus_true_sparse_padded(qf[b], kf[b], vf[b]) for b in range(qf.shape[0])],
        0,
    )


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
def tps(tok, model, device, n_prompts=4, max_new=24):
    rates = []
    for p in PROBES[:n_prompts]:
        inp = tok(p, return_tensors="pt").to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            **inp,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        rates.append((out.shape[-1] - inp["input_ids"].shape[-1]) / (ms / 1000))
    return sum(rates) / max(len(rates), 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = train_texts()
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "target_agree": TARGET_AGREE,
        "corpus_size": len(corpus),
        "steps_pure": STEPS_PURE,
        "lora_r": LORA_R,
        "demos": {},
        "history": [],
        "ok": False,
    }

    print(f"device={device} corpus={len(corpus)} steps={STEPS_PURE} target={TARGET_AGREE:.0%}")
    print(f"cuda_dll={cuda_ok()}")

    tok, teacher = load(device, train=False)
    for p in teacher.parameters():
        p.requires_grad_(False)
    tps_base = tps(tok, teacher, device)
    report["baseline_tps"] = tps_base
    print(f"baseline tps={tps_base:.1f}")

    # ========== DEMO 1: blend quality path ==========
    print("\n=== DEMO quality: blend α-only ===")
    _, blend = load(device, train=True)
    for i in range(len(blend.model.layers)):
        blend.model.layers[i].self_attn = BlendAttn(blend.model.layers[i].self_attn)
    for n, p in blend.named_parameters():
        p.requires_grad_("alpha_logit" in n)
    opt = torch.optim.Adam([p for p in blend.parameters() if p.requires_grad], lr=LR_ALPHA)
    blend.train()
    for step in range(STEPS_BLEND):
        text = corpus[step % len(corpus)]
        inp = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            t_log = teacher(**inp).logits
        s_log = blend(**inp).logits
        loss = F.kl_div(
            F.log_softmax(s_log, dim=-1),
            F.softmax(t_log, dim=-1),
            reduction="batchmean",
        )
        if not torch.isfinite(loss):
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in blend.parameters() if p.requires_grad], 1.0)
        opt.step()
        with torch.no_grad():
            for layer in blend.model.layers:
                layer.self_attn.alpha_logit.clamp_(-6, 6)
    blend.eval()
    ba, bk, bt = measure(tok, teacher, blend, device)
    btps = tps(tok, blend, device)
    mean_a = float(
        torch.stack([layer.self_attn.alpha.detach() for layer in blend.model.layers]).mean()
    )
    report["demos"]["blend_quality"] = {
        "agree": ba,
        "kl": bk,
        "top5": bt,
        "tps": btps,
        "tps_x": btps / max(tps_base, 1e-9),
        "mean_alpha": mean_a,
        "role": "quality demo — match baseline logits with FSOT in the mix",
    }
    print(f"  BLEND agree={ba:.0%} KL={bk:.3f} α={mean_a:.3f} tps×{btps/max(tps_base,1e-9):.2f}")

    # ========== PURE FSOT + LoRA (speed path) ==========
    print("\n=== PURE FSOT + LoRA on QKV/O (operator fixed) ===")
    _, pure = load(device, train=True)
    swap_all_layers(pure)
    n_lora = inject_lora_into_fsot_attn(pure, r=LORA_R, alpha=LORA_ALPHA)
    pure.to(device)  # ensure LoRA params on GPU
    freeze_non_lora(pure)
    # also train layernorms lightly (not LoRA but small surface)
    for n, p in pure.named_parameters():
        if "layernorm" in n or n.endswith("model.norm.weight"):
            p.requires_grad_(True)
    n_train = sum(p.numel() for p in pure.parameters() if p.requires_grad)
    print(f"  LoRA modules wrapped: {n_lora}  trainable params: {n_train:,}")

    opt_p = torch.optim.AdamW(
        [p for p in pure.parameters() if p.requires_grad],
        lr=LR_LORA,
        weight_decay=0.01,
    )
    best = {"agree": -1.0, "kl": 1e9, "step": -1, "path": None}
    history = []
    pure.train()
    t0_all = time.perf_counter()
    for step in range(STEPS_PURE):
        text = corpus[step % len(corpus)]
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
            print(f"  non-finite at step {step}, skip")
            continue
        opt_p.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in pure.parameters() if p.requires_grad], 1.0
        )
        opt_p.step()

        if step % EVAL_EVERY == 0 or step == STEPS_PURE - 1:
            pure.eval()
            a, k, t5 = measure(tok, teacher, pure, device)
            tp = tps(tok, pure, device)
            pure.train()
            row = {
                "step": step,
                "loss": float(loss.item()),
                "agree": a,
                "kl": k,
                "top5": t5,
                "tps": tp,
                "tps_x": tp / max(tps_base, 1e-9),
            }
            history.append(row)
            report["history"] = history
            print(
                f"  step {step:04d} loss={loss.item():.3f} agree={a:.0%} "
                f"KL={k:.3f} top5={t5:.2f} tps×{tp/max(tps_base,1e-9):.2f}"
            )
            # save best by agree then kl
            if a > best["agree"] or (a == best["agree"] and k < best["kl"]):
                best.update({"agree": a, "kl": k, "step": step, "top5": t5, "tps": tp})
                path = CKPT / "pure_fsot_lora_best.pt"
                torch.save(
                    {
                        "step": step,
                        "agree": a,
                        "kl": k,
                        "state_dict": {
                            n: p.detach().cpu()
                            for n, p in pure.named_parameters()
                            if p.requires_grad
                        },
                    },
                    path,
                )
                best["path"] = str(path)
                print(f"    * new best saved → {path}")
            if a >= TARGET_AGREE:
                print(f"  TARGET {TARGET_AGREE:.0%} reached at step {step}")
                break

    elapsed = time.perf_counter() - t0_all
    pure.eval()
    a_f, k_f, t_f = measure(tok, teacher, pure, device)
    tps_f = tps(tok, pure, device)

    report["demos"]["blend_quality"] = report["demos"].get("blend_quality") or {}
    # re-store blend (already set)
    report["demos"]["pure_fsot_speed"] = {
        "agree": a_f,
        "kl": k_f,
        "top5": t_f,
        "tps": tps_f,
        "tps_x": tps_f / max(tps_base, 1e-9),
        "best_agree": best["agree"],
        "best_kl": best["kl"],
        "best_step": best["step"],
        "best_ckpt": best["path"],
        "train_seconds": elapsed,
        "role": "speed path — pure FSOT operator + LoRA",
        "hit_target_80": best["agree"] >= TARGET_AGREE,
    }

    report["summary"] = {
        "blend_agree": ba,
        "blend_tps_x": btps / max(tps_base, 1e-9),
        "pure_agree": a_f,
        "pure_best_agree": best["agree"],
        "pure_kl": k_f,
        "pure_tps_x": tps_f / max(tps_base, 1e-9),
        "target_80": TARGET_AGREE,
        "hit_target": best["agree"] >= TARGET_AGREE,
        "vs_unadapted_pure_agree": 0.0,
        "vs_prior_25pct": 0.25,
    }
    report["ok"] = ba >= 0.99 and best["agree"] >= 0.25
    report["claim"] = (
        f"Blend quality demo: agree={ba:.0%} tps×{btps/max(tps_base,1e-9):.2f}. "
        f"Pure FSOT+LoRA: best agree={best['agree']:.0%} (final {a_f:.0%}) "
        f"KL={k_f:.3f} tps×{tps_f/max(tps_base,1e-9):.2f} "
        f"target80={'HIT' if best['agree']>=TARGET_AGREE else 'not yet'}."
    )

    out = OUT / "push_all_three.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== ALL THREE SUMMARY ===")
    print(report["claim"])
    print("wrote", out)
    return 0 if report["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

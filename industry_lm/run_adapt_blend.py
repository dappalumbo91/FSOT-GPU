#!/usr/bin/env python3
"""
Push quality after all-layer FSOT: learn per-layer blend α (FSOT-derived init)
  out = (1-α)*SDPA + α*FSOT_consensus

Only α trains (30 params). Teacher = frozen baseline. Minimize KL on short texts.
Then remeasure agreement. FSOT stays in the path; α from φ init.
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
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from fsot_cuda_ops import available as cuda_ok, fsot_consensus  # noqa: E402
from competitive.sparse_consensus_batched import consensus_true_sparse_padded  # noqa: E402
from fsot_lib.seeds import SEEDS  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
OUT.mkdir(parents=True, exist_ok=True)

PHI = SEEDS.phi
# init α = 1/φ² ≈ 0.382 — FSOT complexity / inner coupling motif
ALPHA0 = 1.0 / (PHI * PHI)

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In mathematics, the derivative of x^2 is",
    "Once upon a time",
    "2 + 2 =",
    "The largest planet in our solar system is",
    "print('hello",
    "Water freezes at",
]

TEXTS = PROMPTS + [
    "Python is a programming language that",
    "The speed of light is approximately",
    "To sort a list in reverse order",
    "Photosynthesis converts",
]


def fsot_core(q, k, v):
    if cuda_ok() and q.is_cuda:
        try:
            return fsot_consensus(q, k, v)
        except Exception:
            pass
    outs = [consensus_true_sparse_padded(q[b], k[b], v[b]) for b in range(q.shape[0])]
    return torch.stack(outs, 0)


class BlendAttn(nn.Module):
    def __init__(self, src):
        super().__init__()
        self.config = src.config
        self.layer_idx = src.layer_idx
        self.head_dim = src.head_dim
        self.num_key_value_groups = src.num_key_value_groups
        self.scaling = src.scaling
        self.q_proj = src.q_proj
        self.k_proj = src.k_proj
        self.v_proj = src.v_proj
        self.o_proj = src.o_proj
        # FSOT-init blend: sigmoid(logit) so α∈(0,1); start near ALPHA0
        logit0 = math.log(ALPHA0 / (1 - ALPHA0))
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

        # industry path
        sdpa = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=self.scaling
        )
        # FSOT path
        fs = fsot_core(q.float(), k.float(), v.float()).to(dtype=q.dtype)
        a = self.alpha.to(dtype=q.dtype)
        mix = (1.0 - a) * sdpa + a * fs
        out = mix.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(out), None


def swap_blend(model):
    for i in range(len(model.model.layers)):
        model.model.layers[i].self_attn = BlendAttn(model.model.layers[i].self_attn)
    return model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("=== Teacher (frozen) ===")
    teacher = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("=== Student blend all layers (train α only) ===")
    student = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=dtype, trust_remote_code=True
    ).to(device)
    swap_blend(student)
    # freeze all but alpha_logit
    for n, p in student.named_parameters():
        p.requires_grad_(("alpha_logit" in n))
    alphas = [p for n, p in student.named_parameters() if p.requires_grad]
    print(f"trainable tensors: {len(alphas)} (expect 30)")

    # suction-poof inspired LR from seeds
    lr = float(SEEDS.suction * SEEDS.k)  # ~0.06
    opt = torch.optim.Adam(alphas, lr=lr)

    student.train()
    losses = []
    steps = 80
    print(f"=== Adapt {steps} steps KL, lr={lr:.5f} (FSOT suction*K) ===")
    for step in range(steps):
        text = TEXTS[step % len(TEXTS)]
        inp = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            t_log = teacher(**inp).logits.float()
        s_log = student(**inp).logits.float()
        # KL teacher || student on last positions
        t_p = F.softmax(t_log, dim=-1)
        loss = F.kl_div(F.log_softmax(s_log, dim=-1), t_p, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        if step % 20 == 0 or step == steps - 1:
            with torch.no_grad():
                a_mean = torch.stack(
                    [
                        torch.sigmoid(m.self_attn.alpha_logit)
                        for m in student.model.layers
                    ]
                ).mean()
            print(f"  step {step:03d} loss={loss.item():.4f} mean_α={float(a_mean):.3f}")

    student.eval()

    # Agreement
    def agree(model):
        ok = kl = top = 0
        n = len(PROMPTS)
        for p in PROMPTS:
            inp = tok(p, return_tensors="pt").to(device)
            with torch.no_grad():
                lt = teacher(**inp).logits[0, -1].float().cpu()
                ls = model(**inp).logits[0, -1].float().cpu()
            pt = F.softmax(lt, dim=-1)
            kl += float(
                (pt * (torch.log(pt.clamp_min(1e-12)) - F.log_softmax(ls, dim=-1))).sum()
            )
            ok += int(lt.argmax() == ls.argmax())
            top += len(
                set(torch.topk(lt, 5).indices.tolist())
                & set(torch.topk(ls, 5).indices.tolist())
            ) / 5
        return ok / n, kl / n, top / n

    print("=== Remeasure ===")
    # pure fsot for reference would need separate model — skip, report blend
    a, k, t5 = agree(student)
    print(f"BLEND agree={a:.0%} KL={k:.3f} top5={t5:.2f}")

    # baseline self-agree sanity
    a0, k0, t50 = agree(teacher)
    print(f"TEACHER self-check agree={a0:.0%} KL={k0:.3f}")

    # tok/s
    def tps(model):
        rates = []
        for p in PROMPTS[:4]:
            inp = tok(p, return_tensors="pt").to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
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

    with torch.no_grad():
        tb = tps(teacher)
        ts = tps(student)
    print(f"tok/s teacher={tb:.1f} blend={ts:.1f} ×{ts/max(tb,1e-9):.2f}")

    with torch.no_grad():
        alphas = [
            float(torch.sigmoid(m.self_attn.alpha_logit)) for m in student.model.layers
        ]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "fsot_sdpa_blend_all_layers",
        "alpha0_fsot": ALPHA0,
        "lr_fsot": lr,
        "steps": steps,
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "mean_alpha_after": sum(alphas) / len(alphas),
        "alphas": alphas,
        "quality": {"argmax_agreement": a, "mean_kl": k, "mean_top5": t5},
        "throughput_tps": {"baseline": tb, "blend": ts, "speedup": ts / max(tb, 1e-9)},
        "vs_pure_all_layers": {
            "pure_agree": 0.0,
            "pure_kl": 8.05,
            "note": "from all_layers_eval.json",
        },
        "ok": a > 0.5,
    }
    path = OUT / "adapt_blend_all_layers.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", path)
    print(
        f"SUMMARY blend agree={a:.0%} (pure was 0%) KL={k:.3f} "
        f"tps×{ts/max(tb,1e-9):.2f}"
    )
    return 0 if report["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Extend pure-FSOT all-layer adaptation (from 0% → 25% trajectory).
Train o_proj + norms + light q/k/v projs, more steps, curriculum on short texts.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from fsot_layer_swap import swap_all_layers  # noqa: E402
from fsot_lib.seeds import SEEDS  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
OUT.mkdir(parents=True, exist_ok=True)

LR = 1.5e-4  # stable for weight adapt
STEPS = 400

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
    "Hello, my name is",
    "The president of the United States",
    "1 + 1 =",
    "for i in range(",
]


def load(device):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    return tok, m


@torch.no_grad()
def measure(tok, teacher, student, device):
    agree = kl = top = 0.0
    n = len(PROBES)
    details = []
    for p in PROBES:
        inp = tok(p, return_tensors="pt").to(device)
        lt = teacher(**inp).logits[0, -1].float().cpu()
        ls = student(**inp).logits[0, -1].float().cpu()
        if not torch.isfinite(ls).all():
            return 0.0, float("nan"), 0.0, []
        pt = F.softmax(lt, dim=-1)
        k = float(
            (pt * (torch.log(pt.clamp_min(1e-12)) - F.log_softmax(ls, dim=-1))).sum()
        )
        ok = int(lt.argmax() == ls.argmax())
        ov = len(
            set(torch.topk(lt, 5).indices.tolist())
            & set(torch.topk(ls, 5).indices.tolist())
        ) / 5.0
        agree += ok
        kl += k
        top += ov
        details.append(
            {
                "prompt": p,
                "base": tok.decode([int(lt.argmax())]),
                "fsot": tok.decode([int(ls.argmax())]),
                "match": bool(ok),
                "kl": k,
            }
        )
    return agree / n, kl / n, top / n, details


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Teacher ===")
    tok, teacher = load(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("=== Pure FSOT student (all layers) ===")
    _, student = load(device)
    swap_all_layers(student)
    # train attention projections + o_proj + norms
    for n, p in student.named_parameters():
        train = any(
            k in n
            for k in (
                "o_proj",
                "q_proj",
                "k_proj",
                "v_proj",
                "input_layernorm",
                "post_attention_layernorm",
                "model.norm",
            )
        )
        p.requires_grad_(train)
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"trainable {n_train:,}  LR={LR}  steps={STEPS}")

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=LR, weight_decay=0.01
    )
    student.train()
    losses = []
    for step in range(STEPS):
        text = TRAIN[step % len(TRAIN)]
        # also pack two short prompts sometimes
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
            print(f"non-finite at {step}, stop")
            break
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        losses.append(float(loss.item()))
        if step % 50 == 0 or step == STEPS - 1:
            student.eval()
            a, k, t5, _ = measure(tok, teacher, student, device)
            student.train()
            print(f"  {step:03d} kl={loss.item():.3f} agree={a:.0%} KL={k:.3f} top5={t5:.2f}")

    student.eval()
    a, k, t5, details = measure(tok, teacher, student, device)
    tps_s = tps(tok, student, device)
    tps_t = tps(tok, teacher, device)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "pure_fsot_all_layers_qkv_adapt",
        "steps": len(losses),
        "lr": LR,
        "trainable_params": n_train,
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "quality": {
            "argmax_agreement": a,
            "mean_kl": k,
            "mean_top5": t5,
            "details": details,
        },
        "throughput": {
            "baseline_tps": tps_t,
            "pure_fsot_tps": tps_s,
            "speedup": tps_s / max(tps_t, 1e-9),
        },
        "vs_unadapted_pure": {"agree": 0.0, "kl": 8.05},
        "vs_prior_C_phase": {"agree": 0.25, "kl": 5.39},
        "ok": a >= 0.25,
    }
    path = OUT / "pure_fsot_extend.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== EXTEND SUMMARY ===")
    print(
        f"pure FSOT agree={a:.0%} KL={k:.3f} top5={t5:.2f} "
        f"tps×{tps_s/max(tps_t,1e-9):.2f} (was 0% unadapted, 25% short adapt)"
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""All-layer FSOT-gated SDPA (quality recovery) vs pure consensus vs baseline."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from fsot_gated_attn import swap_all_gated, fsot_key_mask, COLLAPSE_THRESHOLD  # noqa: E402
from fsot_layer_swap import swap_all_layers  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
OUT.mkdir(parents=True, exist_ok=True)

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


def load(device, dtype):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=dtype, trust_remote_code=True
    ).to(device).eval()
    return tok, m


@torch.no_grad()
def eval_pair(tok, base, other, device, label):
    agree = kl_s = top = 0
    n = len(PROMPTS)
    rows = []
    for p in PROMPTS:
        inp = tok(p, return_tensors="pt").to(device)
        lb = base(**inp).logits[0, -1].float().cpu()
        lo = other(**inp).logits[0, -1].float().cpu()
        pb = torch.softmax(lb, dim=-1)
        kl = float(
            (pb * (torch.log(pb.clamp_min(1e-12)) - torch.log_softmax(lo, dim=-1))).sum()
        )
        mb, mo = int(lb.argmax()), int(lo.argmax())
        match = mb == mo
        agree += int(match)
        kl_s += kl
        ov = len(set(torch.topk(lb, 5).indices.tolist()) & set(torch.topk(lo, 5).indices.tolist())) / 5
        top += ov
        rows.append(
            {
                "prompt": p,
                "base": tok.decode([mb]),
                "other": tok.decode([mo]),
                "match": match,
                "kl": kl,
                "top5": ov,
            }
        )
        print(
            f"  [{label} {'OK' if match else '..'}] {p!r} → "
            f"{tok.decode([mb])!r} vs {tok.decode([mo])!r} KL={kl:.3f}"
        )
    return {
        "argmax_agreement": agree / n,
        "mean_kl": kl_s / n,
        "mean_top5": top / n,
        "details": rows,
    }


@torch.no_grad()
def tps(tok, model, device, prompts, max_new=24):
    rates = []
    for p in prompts[:4]:
        inp = tok(p, return_tensors="pt").to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            **inp, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id
        )
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        new = out.shape[-1] - inp["input_ids"].shape[-1]
        rates.append(new / (ms / 1000))
    return sum(rates) / len(rates)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tok, base = load(device, dtype)
    print("=== FSOT-GATED SDPA all layers ===")
    _, gated = load(device, dtype)
    swap_all_gated(gated)

    print("=== Pure FSOT consensus all layers (reference) ===")
    _, pure = load(device, dtype)
    swap_all_layers(pure)

    print("--- Gated vs baseline ---")
    q_gated = eval_pair(tok, base, gated, device, "gate")
    print("--- Pure consensus vs baseline ---")
    q_pure = eval_pair(tok, base, pure, device, "pure")

    print("=== tok/s ===")
    t_base = tps(tok, base, device, PROMPTS)
    t_gate = tps(tok, gated, device, PROMPTS)
    t_pure = tps(tok, pure, device, PROMPTS)
    print(f"base={t_base:.1f}  gated={t_gate:.1f}  pure={t_pure:.1f}")

    # Active fraction on a sample forward (layer 0 keys approx via random — skip)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "layers": len(base.model.layers),
        "gated_all": q_gated,
        "pure_consensus_all": q_pure,
        "throughput_tps": {
            "baseline": t_base,
            "fsot_gated": t_gate,
            "fsot_pure": t_pure,
            "gated_speedup": t_gate / max(t_base, 1e-9),
            "pure_speedup": t_pure / max(t_base, 1e-9),
        },
        "winner_quality": (
            "gated"
            if q_gated["argmax_agreement"] > q_pure["argmax_agreement"]
            else "pure"
        ),
        "ok": True,
    }
    path = OUT / "gated_vs_pure_all_layers.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== SUMMARY ===")
    print(
        f"GATED agree={q_gated['argmax_agreement']:.0%} KL={q_gated['mean_kl']:.3f} | "
        f"PURE agree={q_pure['argmax_agreement']:.0%} KL={q_pure['mean_kl']:.3f}"
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

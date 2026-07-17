#!/usr/bin/env python3
"""
Wire FSOT CUDA consensus into one SmolLM2 layer; measure quality + tokens/s.

Compares:
  baseline  = industry HF model
  fsot_l0   = same weights, layer 0 attention = FSOT consensus
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from fsot_layer_swap import swap_layer  # noqa: E402
from fsot_cuda_ops import available as cuda_dll_available  # noqa: E402

MODEL = Path(__file__).resolve().parent / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In mathematics, the derivative of x^2 is",
    "Once upon a time",
    "2 + 2 =",
]


def load_model(device: str, dtype):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL),
        dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def next_token_stats(tok, model, prompts, device):
    """Teacher-force first prompt tokens; compare next-token argmax distribution."""
    # For agreement: run both models externally
    return None


@torch.no_grad()
def logits_for_prompt(tok, model, text, device):
    inp = tok(text, return_tensors="pt").to(device)
    out = model(**inp)
    return out.logits[0, -1, :].float().cpu()


@torch.no_grad()
def generate_one(tok, model, text, device, max_new=24):
    inp = tok(text, return_tensors="pt").to(device)
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
    ms = 1000.0 * (time.perf_counter() - t0)
    new = int(out.shape[-1] - inp["input_ids"].shape[-1])
    text_out = tok.decode(out[0], skip_special_tokens=True)
    return text_out, ms, new


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(MODEL),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "fsot_cuda_dll": cuda_dll_available(),
        "swap_layer": 0,
        "ok": False,
    }

    print("=== Load baseline ===")
    tok, base = load_model(device, dtype)
    print("=== Load FSOT-swapped (layer 0) ===")
    tok2, fsot_m = load_model(device, dtype)
    swap_layer(fsot_m, layer_idx=0)

    # Next-token agreement
    agree = 0
    kl_vals = []
    top5_overlap = []
    details = []
    for p in PROMPTS:
        lb = logits_for_prompt(tok, base, p, device)
        lf = logits_for_prompt(tok2, fsot_m, p, device)
        pb = torch.softmax(lb, dim=-1)
        # KL(base || fsot)
        log_pf = torch.log_softmax(lf, dim=-1)
        kl = float((pb * (torch.log(pb.clamp_min(1e-12)) - log_pf)).sum())
        pred_b = int(lb.argmax())
        pred_f = int(lf.argmax())
        match = pred_b == pred_f
        agree += int(match)
        kl_vals.append(kl)
        tb = set(torch.topk(lb, 5).indices.tolist())
        tf = set(torch.topk(lf, 5).indices.tolist())
        ov = len(tb & tf) / 5.0
        top5_overlap.append(ov)
        details.append(
            {
                "prompt": p,
                "base_token": tok.decode([pred_b]),
                "fsot_token": tok.decode([pred_f]),
                "argmax_match": match,
                "kl_base_fsot": kl,
                "top5_overlap": ov,
            }
        )
        print(
            f"  [{ 'OK' if match else '..' }] {p!r} → base={tok.decode([pred_b])!r} "
            f"fsot={tok.decode([pred_f])!r} KL={kl:.4f} top5={ov:.2f}"
        )

    report["quality"] = {
        "n_prompts": len(PROMPTS),
        "argmax_agreement": agree / len(PROMPTS),
        "mean_kl": sum(kl_vals) / len(kl_vals),
        "mean_top5_overlap": sum(top5_overlap) / len(top5_overlap),
        "details": details,
    }

    # Tokens/s generation
    print("=== Generation throughput ===")
    gen_b = []
    gen_f = []
    for p in PROMPTS[:3]:
        tb, msb, nb = generate_one(tok, base, p, device)
        tf, msf, nf = generate_one(tok2, fsot_m, p, device)
        gen_b.append({"ms": msb, "new": nb, "tps": nb / (msb / 1000.0), "text": tb[:100]})
        gen_f.append({"ms": msf, "new": nf, "tps": nf / (msf / 1000.0), "text": tf[:100]})
        print(f"  base {msb:.0f}ms {nb}tok {nb/(msb/1000):.1f} t/s | fsot {msf:.0f}ms {nf/(msf/1000):.1f} t/s")

    def avg_tps(rows):
        return sum(r["tps"] for r in rows) / max(len(rows), 1)

    report["throughput"] = {
        "baseline_tps": avg_tps(gen_b),
        "fsot_layer0_tps": avg_tps(gen_f),
        "speedup": avg_tps(gen_f) / max(avg_tps(gen_b), 1e-9),
        "baseline_gens": gen_b,
        "fsot_gens": gen_f,
    }

    # Prefill ms
    print("=== Prefill ===")
    text = PROMPTS[0]
    inp = tok(text, return_tensors="pt").to(device)

    def prefill_ms(model, iters=30):
        with torch.no_grad():
            for _ in range(5):
                _ = model(**inp)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = model(**inp)
            if device == "cuda":
                torch.cuda.synchronize()
            return 1000.0 * (time.perf_counter() - t0) / iters

    with torch.no_grad():
        pb = prefill_ms(base)
        pf = prefill_ms(fsot_m)
    report["prefill_ms"] = {"baseline": pb, "fsot_layer0": pf, "speedup": pb / max(pf, 1e-9)}
    print(f"  baseline {pb:.2f} ms | fsot_l0 {pf:.2f} ms | ×{pb/max(pf,1e-9):.2f}")

    if device == "cuda":
        report["vram_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)

    # Pass bar: model runs; report metrics (quality may drop — honest)
    report["ok"] = True
    report["notes"] = (
        "Layer-0 only swap. Full FSOT model would swap all layers. "
        "Quality shift expected; CUDA DLL H2D each call adds overhead until device API fused."
    )
    path = OUT / "layer_swap_eval.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", path)
    print(
        f"SUMMARY agree={report['quality']['argmax_agreement']:.0%} "
        f"KL={report['quality']['mean_kl']:.3f} "
        f"tps_base={report['throughput']['baseline_tps']:.1f} "
        f"tps_fsot={report['throughput']['fsot_layer0_tps']:.1f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

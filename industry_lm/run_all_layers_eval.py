#!/usr/bin/env python3
"""
Swap ALL SmolLM2 attention layers to FSOT consensus; remeasure quality + speed.

baseline  = industry HF
fsot_all  = same SafeTensors weights, every self_attn = FSOT CUDA consensus
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
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from fsot_layer_swap import swap_all_layers  # noqa: E402
from fsot_cuda_ops import available as cuda_dll_available  # noqa: E402

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


def load_model(device: str, dtype):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=dtype, trust_remote_code=True
    )
    model.to(device)
    model.eval()
    return tok, model


@torch.no_grad()
def logits_last(tok, model, text, device):
    inp = tok(text, return_tensors="pt").to(device)
    return model(**inp).logits[0, -1, :].float().cpu()


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
    return tok.decode(out[0], skip_special_tokens=True), ms, new


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("=== Baseline ===")
    tok, base = load_model(device, dtype)
    n_layers = len(base.model.layers)
    print(f"layers={n_layers}  cuda_dll={cuda_dll_available()}")

    print("=== FSOT all-layer swap ===")
    tok2, fsot = load_model(device, dtype)
    swap_all_layers(fsot)
    n_swapped = sum(
        1
        for i in range(n_layers)
        if type(fsot.model.layers[i].self_attn).__name__ == "FsotLlamaAttention"
    )
    print(f"swapped {n_swapped}/{n_layers}")

    # Quality
    print("=== Next-token agreement ===")
    agree = 0
    kl_vals = []
    top5 = []
    details = []
    for p in PROMPTS:
        lb = logits_last(tok, base, p, device)
        lf = logits_last(tok2, fsot, p, device)
        pb = torch.softmax(lb, dim=-1)
        log_pf = torch.log_softmax(lf, dim=-1)
        kl = float((pb * (torch.log(pb.clamp_min(1e-12)) - log_pf)).sum())
        pb_i = int(lb.argmax())
        pf_i = int(lf.argmax())
        match = pb_i == pf_i
        agree += int(match)
        kl_vals.append(kl)
        ov = len(
            set(torch.topk(lb, 5).indices.tolist())
            & set(torch.topk(lf, 5).indices.tolist())
        ) / 5.0
        top5.append(ov)
        details.append(
            {
                "prompt": p,
                "base": tok.decode([pb_i]),
                "fsot": tok.decode([pf_i]),
                "match": match,
                "kl": kl,
                "top5_overlap": ov,
            }
        )
        print(
            f"  [{'OK' if match else '..'}] {p!r} → {tok.decode([pb_i])!r} vs "
            f"{tok.decode([pf_i])!r}  KL={kl:.3f} top5={ov:.2f}"
        )

    # Throughput
    print("=== Generation tok/s ===")
    gen_b, gen_f = [], []
    for p in PROMPTS[:4]:
        tb, msb, nb = generate_one(tok, base, p, device)
        tf, msf, nf = generate_one(tok2, fsot, p, device)
        gen_b.append({"ms": msb, "new": nb, "tps": nb / (msb / 1000), "text": tb[:120]})
        gen_f.append({"ms": msf, "new": nf, "tps": nf / (msf / 1000), "text": tf[:120]})
        print(
            f"  base {msb:.0f}ms {nb/(msb/1000):.1f} t/s | "
            f"fsot_all {msf:.0f}ms {nf/(msf/1000):.1f} t/s"
        )

    def avg(xs, key):
        return sum(x[key] for x in xs) / max(len(xs), 1)

    # Prefill
    inp = tok(PROMPTS[0], return_tensors="pt").to(device)

    def prefill(model, iters=20):
        with torch.no_grad():
            for _ in range(3):
                _ = model(**inp)
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = model(**inp)
            if device == "cuda":
                torch.cuda.synchronize()
            return 1000 * (time.perf_counter() - t0) / iters

    with torch.no_grad():
        pb = prefill(base)
        pf = prefill(fsot)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": str(MODEL),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "fsot_cuda_dll": cuda_dll_available(),
        "layers_total": n_layers,
        "layers_swapped": n_swapped,
        "mode": "all_layers",
        "quality": {
            "n_prompts": len(PROMPTS),
            "argmax_agreement": agree / len(PROMPTS),
            "mean_kl": sum(kl_vals) / len(kl_vals),
            "mean_top5_overlap": sum(top5) / len(top5),
            "details": details,
        },
        "throughput": {
            "baseline_tps": avg(gen_b, "tps"),
            "fsot_all_tps": avg(gen_f, "tps"),
            "speedup": avg(gen_f, "tps") / max(avg(gen_b, "tps"), 1e-9),
            "baseline_gens": gen_b,
            "fsot_gens": gen_f,
        },
        "prefill_ms": {
            "baseline": pb,
            "fsot_all": pf,
            "speedup": pb / max(pf, 1e-9),
        },
        "ok": True,
        "notes": (
            "All attention layers use FSOT consensus; weights not retrained. "
            "Quality shift is expected until FSOT-LR/LoRA adaptation."
        ),
    }
    if device == "cuda":
        report["vram_mib"] = round(torch.cuda.max_memory_allocated() / 1024**2, 2)

    path = OUT / "all_layers_eval.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== SUMMARY (ALL LAYERS) ===")
    print(
        f"agree={report['quality']['argmax_agreement']:.0%}  "
        f"KL={report['quality']['mean_kl']:.3f}  "
        f"top5={report['quality']['mean_top5_overlap']:.2f}"
    )
    print(
        f"tps base={report['throughput']['baseline_tps']:.1f}  "
        f"fsot_all={report['throughput']['fsot_all_tps']:.1f}  "
        f"×{report['throughput']['speedup']:.2f}"
    )
    print(
        f"prefill base={pb:.2f}ms  fsot_all={pf:.2f}ms  ×{pb/max(pf,1e-9):.2f}"
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

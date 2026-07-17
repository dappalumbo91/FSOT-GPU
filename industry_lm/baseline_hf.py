#!/usr/bin/env python3
"""Industry baseline: Hugging Face transformers + SafeTensors on GPU."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT.parent / "results" / "industry_lm"
OUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In fluid spacetime, observation",
]


def run_baseline(model_dir: Path | None = None, max_new_tokens: int = 32) -> dict[str, Any]:
    model_dir = Path(model_dir or DEFAULT_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    report: dict[str, Any] = {
        "path": "industry_hf_transformers",
        "model_dir": str(model_dir),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "dtype": str(dtype),
        "params": sum(p.numel() for p in model.parameters()),
        "generations": [],
        "latency": {},
    }

    # Prefill latency on fixed tokens
    text = PROMPTS[0]
    inputs = tok(text, return_tensors="pt").to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(20):
            _ = model(**inputs)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    report["latency"]["prefill_20x_ms"] = 1000.0 * dt / 20
    report["latency"]["input_tokens"] = int(inputs["input_ids"].numel())

    # Generate smoke
    for p in PROMPTS:
        inp = tok(p, return_tensors="pt").to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        if device == "cuda":
            torch.cuda.synchronize()
        ms = 1000.0 * (time.perf_counter() - t0)
        gen = tok.decode(out[0], skip_special_tokens=True)
        report["generations"].append(
            {
                "prompt": p,
                "output": gen,
                "ms": ms,
                "new_tokens": int(out.shape[-1] - inp["input_ids"].shape[-1]),
            }
        )

    if device == "cuda":
        report["vram_allocated_mib"] = round(torch.cuda.memory_allocated() / 1024**2, 2)

    path = OUT / "baseline_hf.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["wrote"] = str(path)
    return report


if __name__ == "__main__":
    r = run_baseline()
    print(json.dumps({k: r[k] for k in r if k != "generations"}, indent=2))
    for g in r["generations"]:
        print("---")
        print(g["output"][:200])
        print(f"  [{g['ms']:.1f} ms, +{g['new_tokens']} tok]")

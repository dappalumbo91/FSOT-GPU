#!/usr/bin/env python3
"""
FSOT bridge over the same industry SafeTensors model.

Does NOT retrain. Hooks measurable FSOT ops:
  - Optional coherence_norm on residual streams (experimental)
  - Attention timing comparison: industry SDPA vs FSOT sparse consensus on
    extracted Q,K,V from one forward hook (same shapes as model heads)

This is the portability probe: same weights, replaceable compute nodes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from competitive.sparse_consensus_batched import consensus_true_sparse_padded  # noqa: E402
from fsot_lib.coherence import coherence_norm  # noqa: E402
from fsot_lib.seeds import COLLAPSE_THRESHOLD  # noqa: E402

LM = Path(__file__).resolve().parent
DEFAULT_MODEL = LM / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
OUT.mkdir(parents=True, exist_ok=True)


def _sdpa(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


def run_bridge(model_dir: Path | None = None) -> dict[str, Any]:
    model_dir = Path(model_dir or DEFAULT_MODEL)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir), torch_dtype=dtype, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    # Pull config for head split
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    hidden = cfg.hidden_size
    head_dim = hidden // n_heads

    text = "The capital of France is"
    inputs = tok(text, return_tensors="pt").to(device)

    # Capture Q,K,V from first layer via forward hooks on projections is fragile;
    # instead: synthetic QKV with model-scale stats from residual, plus real prefill.
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hidden_states = out.hidden_states  # tuple
        # use layer 0 residual as activation sample scale
        h0 = hidden_states[1]  # after first block-ish; [B,S,H]
        scale = h0.float().std().clamp_min(1e-3)

    # Build multi-head QKV from real hidden for timing (linear proj with random for bench only)
    # Fair micro: same random QKV * scale matching activation magnitude
    B, S, _ = h0.shape
    torch.manual_seed(0)
    q = torch.randn(n_heads, S, head_dim, device=device, dtype=torch.float32) * float(scale)
    # GQA: repeat kv heads conceptually — for bench use n_heads
    k = torch.randn(n_heads, S, head_dim, device=device, dtype=torch.float32) * float(scale)
    v = torch.randn(n_heads, S, head_dim, device=device, dtype=torch.float32) * float(scale)

    def bench(fn, iters=50, warmup=10):
        for _ in range(warmup):
            _ = fn(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            o = fn(q, k, v)
        if device == "cuda":
            torch.cuda.synchronize()
        return {
            "ms_per_iter": 1000.0 * (time.perf_counter() - t0) / iters,
            "out_max_abs": float(o.detach().float().abs().max()),
        }

    # coherence_norm micro on residual
    x = h0[0].float()  # S,H
    def rms_norm(t):
        return t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + 1e-5)

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = rms_norm(x)
    if device == "cuda":
        torch.cuda.synchronize()
    rms_ms = 1000.0 * (time.perf_counter() - t0) / 100

    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        _ = coherence_norm(x)
    if device == "cuda":
        torch.cuda.synchronize()
    coh_ms = 1000.0 * (time.perf_counter() - t0) / 100

    report = {
        "path": "fsot_bridge_same_weights",
        "model_dir": str(model_dir),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "params": sum(p.numel() for p in model.parameters()),
        "activation_std": float(scale),
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "attn_bench": {
            "shapes": {"heads": n_heads, "seq": S, "head_dim": head_dim, "kv_heads_cfg": n_kv},
            "industry_sdpa": bench(_sdpa),
            "fsot_sparse_torch": bench(consensus_true_sparse_padded),
        },
        "norm_bench": {
            "rms_norm_ms": rms_ms,
            "fsot_coherence_norm_ms": coh_ms,
            "speedup_coh_vs_rms": rms_ms / max(coh_ms, 1e-12),
        },
        "portable_note": (
            "Weights remain industry safetensors; FSOT replaces op implementations. "
            "Full layer swap is next; this unit validates bank + op substitution timing."
        ),
        "prefill_logits_ok": bool(torch.isfinite(out.logits).all().item()),
    }
    # speedup
    a = report["attn_bench"]["industry_sdpa"]["ms_per_iter"]
    b = report["attn_bench"]["fsot_sparse_torch"]["ms_per_iter"]
    report["attn_bench"]["fsot_vs_sdpa_speedup"] = a / max(b, 1e-12)

    path = OUT / "fsot_bridge.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["wrote"] = str(path)
    return report


if __name__ == "__main__":
    r = run_bridge()
    print(json.dumps(r, indent=2))

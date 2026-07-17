#!/usr/bin/env python3
"""Prove owned stack works without leaning on industry attention/optimizer APIs."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsot_lib import (
    COLLAPSE_THRESHOLD,
    SEEDS,
    compute_scalar,
    collapse,
    pack_u64,
    unpack_u64,
    coherence_norm,
    suction_poof_lr,
)
from fsot_lib.backend.native_cuda import native_pack_available, run_native_pack_smoke


def main() -> int:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "owned_lib": "fsot_lib",
        "version": "0.1.0",
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "checks": {},
        "ok": True,
    }

    # Pure Python — zero CUDA, zero torch required for core
    S = compute_scalar(D_eff=8.0, observed=True, delta_psi=0.7)
    report["checks"]["scalar_pure"] = {"S": S, "ok": abs(S) > 0}

    codes = [i % 3 for i in range(32)]
    w = pack_u64(codes)
    back = unpack_u64(w)
    report["checks"]["pack_pure"] = {"ok": back == codes, "word": w}

    x = [0.1, 0.95, -0.99, 0.5, 1.2, -1.1]
    y = coherence_norm(x)
    report["checks"]["coherence_pure"] = {
        "ok": len(y) == len(x),
        "out_head": y[:3],
    }

    lr = suction_poof_lr(0, 0.0, 1.0)
    report["checks"]["learn_pure"] = {"lr": lr, "ok": lr > 0}

    # Optional torch backend
    try:
        import torch
        from fsot_lib.consensus import consensus_aggregate, apply_phase_rotation
        from fsot_lib.trinary import pack_u64_torch, unpack_u64_torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        q = torch.randn(8, 16, device=device, dtype=torch.float64)
        k = torch.randn(8, 16, device=device, dtype=torch.float64)
        v = torch.randn(8, 16, device=device, dtype=torch.float64)
        o = consensus_aggregate(q, k, v)
        codes_t = torch.randint(0, 3, (64, 32), device=device, dtype=torch.uint8)
        p = pack_u64_torch(codes_t)
        u = unpack_u64_torch(p)
        report["checks"]["torch_backend"] = {
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "consensus_shape": list(o.shape),
            "pack_roundtrip": bool(torch.equal(codes_t, u)),
            "ok": o.shape == q.shape and bool(torch.equal(codes_t, u)),
        }
    except Exception as e:
        report["checks"]["torch_backend"] = {"ok": False, "error": str(e)}

    # Optional native .cu (our kernel, not cuBLAS)
    if native_pack_available():
        report["checks"]["native_cuda"] = run_native_pack_smoke()
    else:
        report["checks"]["native_cuda"] = {
            "ok": False,
            "reason": "binary not built yet",
        }

    report["ok"] = all(
        c.get("ok", False)
        for name, c in report["checks"].items()
        if name != "native_cuda" or native_pack_available()
    )
    # native must pass if present
    if native_pack_available() and not report["checks"]["native_cuda"].get("ok"):
        report["ok"] = False

    out = ROOT / "results" / "phase2" / "owned_stack_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== FSOT owned stack smoke ===")
    print(f"collapse θ = {COLLAPSE_THRESHOLD:.12g}")
    print(f"pure S     = {S:.12g}")
    for name, c in report["checks"].items():
        print(f"  [{ 'OK' if c.get('ok') else 'FAIL' }] {name}")
    print(f"overall ok = {report['ok']}")
    print(f"wrote {out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

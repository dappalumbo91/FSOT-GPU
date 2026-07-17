#!/usr/bin/env python3
"""
Next competitive goals:
  1) Longer sequence microbench (S=256,512,1024) — sparsity should scale
  2) coherence_norm vs RMSNorm microbench

Uses archive collapse θ; CUDA sparse times when binary present.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fsot_lib.coherence import coherence_norm  # noqa: E402
from fsot_lib.seeds import COLLAPSE_THRESHOLD  # noqa: E402
from competitive.sparse_consensus_batched import consensus_true_sparse_padded  # noqa: E402

OUT = ROOT / "results" / "competitive"
OUT.mkdir(parents=True, exist_ok=True)
CUDA_EXE = ROOT / "phase2_native_gpu" / "cuda" / "fsot_consensus_sparse.exe"


def rms_norm(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def bench(fn, *args, iters=80, warmup=15, device="cuda"):
    for _ in range(warmup):
        y = fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        y = fn(*args)
    if device == "cuda":
        torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - t0) / iters, y


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "long_seq": [],
        "norm": {},
        "ok": False,
    }

    # --- longer seq (torch FSOT sparse vs SDPA) ---
    for S, H, D in [(256, 8, 64), (512, 8, 64), (1024, 8, 64)]:
        torch.manual_seed(0)
        q = torch.randn(H, S, D, device=device, dtype=torch.float32)
        k = torch.randn(H, S, D, device=device, dtype=torch.float32)
        v = torch.randn(H, S, D, device=device, dtype=torch.float32)
        coh = (k.abs() > COLLAPSE_THRESHOLD).float().mean(-1)
        frac = float((coh > 0.5).float().mean().item())

        def sdpa(qq, kk, vv):
            return F.scaled_dot_product_attention(qq, kk, vv, is_causal=True)

        ms_s, _ = bench(sdpa, q, k, v, iters=30, warmup=5, device=device)
        ms_f, _ = bench(
            consensus_true_sparse_padded, q, k, v, iters=30, warmup=5, device=device
        )
        report["long_seq"].append(
            {
                "S": S,
                "H": H,
                "D": D,
                "active_frac": frac,
                "sdpa_ms": ms_s,
                "fsot_torch_sparse_ms": ms_f,
                "speedup_torch": ms_s / max(ms_f, 1e-12),
            }
        )
        print(
            f"S={S}: SDPA {ms_s:.3f} ms | FSOT_torch {ms_f:.3f} ms | "
            f"A/S={frac:.3f} | ×{ms_s/max(ms_f,1e-12):.2f}"
        )

    # --- norm micro ---
    x = torch.randn(2048, 576, device=device, dtype=torch.float32)  # SmolLM hidden
    ms_rms, y1 = bench(rms_norm, x, iters=200, warmup=50, device=device)
    ms_coh, y2 = bench(coherence_norm, x, iters=200, warmup=50, device=device)
    report["norm"] = {
        "shape": list(x.shape),
        "rms_norm_ms": ms_rms,
        "fsot_coherence_norm_ms": ms_coh,
        "speedup_coh_vs_rms": ms_rms / max(ms_coh, 1e-12),
        "rms_out_std": float(y1.std()),
        "coh_out_std": float(y2.float().std()),
        "winner_speed": "fsot_coherence" if ms_coh < ms_rms * 0.95 else (
            "rms" if ms_rms < ms_coh * 0.95 else "tie"
        ),
    }
    print(
        f"norm: RMS {ms_rms:.4f} ms | coh {ms_coh:.4f} ms | "
        f"winner={report['norm']['winner_speed']}"
    )

    # CUDA note: current binary only benches S<=128; extend later
    if CUDA_EXE.is_file():
        r = subprocess.run([str(CUDA_EXE)], capture_output=True, text=True, timeout=60)
        report["cuda_s_le_128"] = r.stdout

    report["ok"] = True
    path = OUT / "long_seq_and_norm.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

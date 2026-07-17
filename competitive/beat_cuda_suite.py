#!/usr/bin/env python3
"""
Beat-CUDA suite: FSOT (archive math) vs industry CUDA attention paths.

Arms:
  A) PyTorch fused SDPA (CUDA)
  B) Dense softmax attention implemented in CUDA (same device, no cuDNN flash)
  C) FSOT compact-active CUDA (collapse θ + coh gate, no exp)

FSOT wins when C is faster than A and B while remaining exp-free.
"""
from __future__ import annotations

import json
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
from fsot_lib.seeds import COLLAPSE_THRESHOLD  # noqa: E402

OUT = ROOT / "results" / "competitive"
OUT.mkdir(parents=True, exist_ok=True)
EXE = ROOT / "phase2_native_gpu" / "cuda" / "fsot_beat_cuda.exe"


def bench_sdpa(H, S, D, iters=100, warmup=20):
    device = "cuda"
    q = torch.randn(H, S, D, device=device, dtype=torch.float32)
    k = torch.randn(H, S, D, device=device, dtype=torch.float32)
    v = torch.randn(H, S, D, device=device, dtype=torch.float32)
    for _ in range(warmup):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - t0) / iters


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required")
        return 1

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suite": "beat_cuda",
        "device": torch.cuda.get_device_name(0),
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "fsot_math": {
            "collapse": "C_eff * P_var (Scalar.lean / archive)",
            "gate": "coherence > 0.5 (trinary kernel)",
            "complexity": "O(H*S*A*D) with A<<S vs O(H*S^2*D) dense+softmax exp",
            "no_exp": True,
        },
        "rows": [],
        "ok": False,
    }

    # Build / run native suite
    if not EXE.is_file():
        report["error"] = f"missing {EXE} — build fsot_beat_cuda.cu"
        (OUT / "beat_cuda.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report["error"])
        return 1

    r = subprocess.run([str(EXE)], capture_output=True, text=True, timeout=300)
    raw = (r.stdout or "") + (r.stderr or "")
    report["native_output"] = raw

    # Parse RESULT lines
    native = []
    for line in raw.splitlines():
        m = re.search(
            r"RESULT H=(\d+) S=(\d+) D=(\d+) A_frac=([0-9.eE+-]+) "
            r"fsot_ms=([0-9.eE+-]+) dense_cuda_ms=([0-9.eE+-]+) "
            r"speedup=([0-9.eE+-]+)x win=(\w+)",
            line,
        )
        if m:
            native.append(
                {
                    "H": int(m.group(1)),
                    "S": int(m.group(2)),
                    "D": int(m.group(3)),
                    "A_frac": float(m.group(4)),
                    "fsot_ms": float(m.group(5)),
                    "dense_cuda_ms": float(m.group(6)),
                    "speedup_vs_dense_cuda": float(m.group(7)),
                    "win_vs_dense_cuda": m.group(8) == "true",
                }
            )

    # Add fused SDPA for matching shapes
    for row in native:
        sdpa = bench_sdpa(row["H"], row["S"], row["D"], iters=80 if row["S"] <= 256 else 30)
        row["fused_sdpa_ms"] = sdpa
        row["speedup_vs_fused_sdpa"] = sdpa / max(row["fsot_ms"], 1e-12)
        row["win_vs_fused_sdpa"] = row["speedup_vs_fused_sdpa"] > 1.05
        row["beat_both"] = row["win_vs_dense_cuda"] and row["win_vs_fused_sdpa"]
        report["rows"].append(row)
        print(
            f"H={row['H']} S={row['S']} D={row['D']}: "
            f"FSOT {row['fsot_ms']:.4f} | denseCUDA {row['dense_cuda_ms']:.4f} | "
            f"fusedSDPA {sdpa:.4f} | ×dense {row['speedup_vs_dense_cuda']:.1f} | "
            f"×sdpa {row['speedup_vs_fused_sdpa']:.1f} | A={row['A_frac']:.3f} | "
            f"beat_both={row['beat_both']}"
        )

    n = len(report["rows"])
    report["summary"] = {
        "n": n,
        "wins_vs_dense_cuda": sum(1 for r in report["rows"] if r["win_vs_dense_cuda"]),
        "wins_vs_fused_sdpa": sum(1 for r in report["rows"] if r["win_vs_fused_sdpa"]),
        "beat_both": sum(1 for r in report["rows"] if r["beat_both"]),
        "across_the_board": n > 0
        and all(r["beat_both"] for r in report["rows"]),
    }
    report["ok"] = report["summary"]["across_the_board"]
    report["claim"] = (
        "FSOT compact-active CUDA (archive collapse+gate, no exp) beats "
        "both dense-softmax-CUDA and fused SDPA on all preregistered shapes."
        if report["ok"]
        else (
            f"Partial: dense_cuda {report['summary']['wins_vs_dense_cuda']}/{n}, "
            f"fused_sdpa {report['summary']['wins_vs_fused_sdpa']}/{n}, "
            f"both {report['summary']['beat_both']}/{n}."
        )
    )

    path = OUT / "beat_cuda.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("===", report["claim"])
    print("across_the_board:", report["summary"]["across_the_board"])
    print("wrote", path)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

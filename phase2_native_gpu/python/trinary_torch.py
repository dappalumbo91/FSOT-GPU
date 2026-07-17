#!/usr/bin/env python3
"""
Phase 2 bridge preview — pure PyTorch trinary pack/unpack (spec-faithful).
CUDA C++ kernels replace this hot path once compiled; this validates the contract.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "phase0"
RESULTS.mkdir(parents=True, exist_ok=True)


def pack_trinary(codes: torch.Tensor) -> torch.Tensor:
    """codes: uint8 [..., 32] values in {0,1,2} → uint64 [...] packed."""
    assert codes.shape[-1] == 32
    codes = codes.to(torch.int64) & 0x3
    shifts = torch.arange(32, device=codes.device, dtype=torch.int64) * 2
    return (codes << shifts).sum(dim=-1).to(torch.int64)


def unpack_trinary(packed: torch.Tensor) -> torch.Tensor:
    """packed: int64 [...] → uint8 [..., 32]."""
    shifts = torch.arange(32, device=packed.device, dtype=torch.int64) * 2
    return ((packed.unsqueeze(-1) >> shifts) & 0x3).to(torch.uint8)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = 8192
    codes = torch.randint(0, 3, (n, 32), device=device, dtype=torch.uint8)
    packed = pack_trinary(codes)
    back = unpack_trinary(packed)
    ok = bool(torch.equal(codes, back))
    # Density: 32 states in 8 bytes vs 32 bytes if uint8 → 4× compression
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "groups": n,
        "roundtrip_ok": ok,
        "bytes_unpacked": n * 32,
        "bytes_packed": n * 8,
        "compression_ratio": 4.0,
        "note": "PyTorch reference; CUDA skeleton in phase2_native_gpu/cuda/trinary_pack.cu",
    }
    path = RESULTS / "trinary_pack_torch.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== FSOT Trinary Pack (PyTorch) ===")
    print(f"device={device} groups={n} roundtrip_ok={ok}")
    print(f"compression 4× ({n*32} → {n*8} bytes)")
    print(f"Wrote {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

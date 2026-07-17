"""
Native CUDA adapter — our .cu kernels, not cuBLAS.

Build: scripts/build_cuda_kernels.ps1
Binary: phase2_native_gpu/cuda/trinary_pack_test.exe
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
KERNEL_EXE = ROOT / "phase2_native_gpu" / "cuda" / "trinary_pack_test.exe"


def native_pack_available() -> bool:
    return KERNEL_EXE.is_file()


def run_native_pack_smoke(timeout_s: float = 60.0) -> dict:
    if not KERNEL_EXE.is_file():
        return {"ok": False, "reason": "kernel binary missing — run build_cuda_kernels.ps1"}
    r = subprocess.run(
        [str(KERNEL_EXE)],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        cwd=str(KERNEL_EXE.parent),
    )
    out = (r.stdout or "") + (r.stderr or "")
    return {
        "ok": r.returncode == 0 and "ok=true" in out,
        "returncode": r.returncode,
        "output": out.strip(),
        "exe": str(KERNEL_EXE),
    }

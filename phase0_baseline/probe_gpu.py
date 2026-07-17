#!/usr/bin/env python3
"""Phase 0 — GPU / CUDA / PyTorch baseline probe for FSOT Formal-GPU experiment."""

from __future__ import annotations

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "phase0"
RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> int:
    report: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": {
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "torch": {},
        "device": {},
        "benchmarks": {},
        "ok": False,
    }

    try:
        import torch
    except ImportError as e:
        report["error"] = f"torch import failed: {e}"
        _write(report)
        print(json.dumps(report, indent=2))
        return 1

    report["torch"] = {
        "version": torch.__version__,
        "cuda_compiled": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": getattr(torch.backends.cudnn, "version", lambda: None)(),
    }

    if not torch.cuda.is_available():
        report["error"] = "CUDA not available to PyTorch"
        _write(report)
        print(json.dumps(report, indent=2))
        return 1

    props = torch.cuda.get_device_properties(0)
    report["device"] = {
        "name": torch.cuda.get_device_name(0),
        "index": 0,
        "total_memory_bytes": props.total_memory,
        "total_memory_mib": round(props.total_memory / (1024**2), 2),
        "major": props.major,
        "minor": props.minor,
        "multi_processor_count": props.multi_processor_count,
        "capability": list(torch.cuda.get_device_capability(0)),
    }

    # Warmup
    a = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    b = torch.randn(1024, 1024, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    for _ in range(5):
        _ = a @ b
    torch.cuda.synchronize()

    # Matmul benchmark (fp32)
    sizes = [1024, 2048, 4096]
    matmul = {}
    for n in sizes:
        x = torch.randn(n, n, device="cuda", dtype=torch.float32)
        y = torch.randn(n, n, device="cuda", dtype=torch.float32)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 20 if n <= 2048 else 10
        for _ in range(iters):
            z = x @ y
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        # 2*n^3 FLOPs per matmul
        flops = 2.0 * (n**3) * iters
        tflops = (flops / dt) / 1e12
        matmul[str(n)] = {
            "iters": iters,
            "seconds": round(dt, 6),
            "approx_tflops": round(tflops, 3),
            "result_norm": float(z.norm().item()),
        }
        del x, y, z
        torch.cuda.empty_cache()

    report["benchmarks"]["matmul_fp32"] = matmul

    # Bandwidth-ish: host->device copy
    host = torch.randn(64 * 1024 * 1024, dtype=torch.float32)  # 256 MiB
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    dev = host.cuda()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    nbytes = host.numel() * 4
    report["benchmarks"]["h2d_copy"] = {
        "bytes": nbytes,
        "seconds": round(dt, 6),
        "gib_per_s": round((nbytes / dt) / (1024**3), 3),
    }
    del host, dev
    torch.cuda.empty_cache()

    # Memory alloc stress (leave headroom for display)
    try:
        free_before, total = torch.cuda.mem_get_info(0)
        # Allocate ~50% of free memory
        n_float = int((free_before * 0.5) // 4)
        n_float = max(n_float - (n_float % 1024), 1024 * 1024)
        blob = torch.empty(n_float, device="cuda", dtype=torch.float32)
        free_after, _ = torch.cuda.mem_get_info(0)
        report["benchmarks"]["alloc_half_free"] = {
            "elements": n_float,
            "mib": round(n_float * 4 / (1024**2), 2),
            "free_before_mib": round(free_before / (1024**2), 2),
            "free_after_mib": round(free_after / (1024**2), 2),
            "ok": True,
        }
        del blob
        torch.cuda.empty_cache()
    except Exception as e:
        report["benchmarks"]["alloc_half_free"] = {"ok": False, "error": str(e)}

    report["ok"] = True
    path = _write(report)
    print("=== FSOT Phase 0 GPU Probe ===")
    print(f"Device: {report['device']['name']}")
    print(f"VRAM:   {report['device']['total_memory_mib']} MiB")
    print(f"CC:     {report['device']['capability']}")
    print(f"Torch:  {report['torch']['version']} (CUDA {report['torch']['cuda_compiled']})")
    print("Matmul FP32 approx TFLOPS:")
    for k, v in matmul.items():
        print(f"  {k}x{k}: {v['approx_tflops']} TFLOPS")
    print(f"H2D:    {report['benchmarks']['h2d_copy']['gib_per_s']} GiB/s")
    print(f"Wrote:  {path}")
    return 0


def _write(report: dict) -> Path:
    path = RESULTS / "gpu_probe.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())

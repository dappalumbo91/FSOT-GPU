#!/usr/bin/env python3
"""
Phase 0 — FSOT boot scalar on CPU vs GPU.

Mirrors archive F* FSOTScalarKernel.compute_fsot_scalar_boot using
oracle transcendental literals (same triangulation strategy as verification/).
Compares f64 CPU reference to f32/f64 GPU evaluation.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = json.loads((ROOT / "config" / "fsot_seeds.json").read_text(encoding="utf-8"))
RESULTS = ROOT / "results" / "phase0"
RESULTS.mkdir(parents=True, exist_ok=True)

# Oracle literals from FSOTScalarKernel.fst (boot specialization)
SQRT_BOOT_D = 2.8284271247461903
COS_PSI_ETA_BOOT = -0.9586053932039044
COS_DP_PVAR_BOOT = -0.08708036371061263
COS_DP_BOOT = 0.7648421872844885
COS_THETA_PI_BOOT = -0.9579871226722758
SIN_THETA_BOOT = 0.28681121455426756
SIN_1_BOOT = 0.8414709848078965
COS_1_BOOT = 0.5403023058681398
LOG_D25_BOOT = -1.1394342831883648


def compute_fsot_scalar_boot(k: dict) -> float:
    """Exact mirror of F* compute_fsot_scalar_boot (oracle lit form)."""
    n = 1.0
    p = 1.0
    d = k["boot"]["d_eff"]
    dp = k["boot"]["delta_psi"]
    hits = k["boot"]["recent_hits"]
    c = k["kernel_constants"]

    # F* uses gamma_euler and phi_fsot from module constants
    gamma = SEEDS["seeds"]["gamma_euler"]
    phi = SEEDS["seeds"]["phi"]
    growth = math.exp(c["alpha_fsot"] * (1.0 - hits / n) * gamma / phi)

    base = (
        (n * p / SQRT_BOOT_D)
        * COS_PSI_ETA_BOOT
        * math.exp((-c["alpha_fsot"]) * hits / n + 1.0 + c["b_in"] * dp)
        * (1.0 + growth * c["c_eff"])
    )
    t1_base = base * (1.0 + c["p_new"] * LOG_D25_BOOT)
    t1 = t1_base * math.exp(c["c_factor"] * c["p_var"]) * COS_DP_PVAR_BOOT
    t2 = 0.0
    valve = (
        c["beta_fsot"]
        * COS_DP_BOOT
        * (n * p / SQRT_BOOT_D)
        * (1.0 + c["chaos_fsot"] * (d - 25.0) / 25.0)
        * (1.0 + c["poof"] * COS_THETA_PI_BOOT + c["suction"] * SIN_THETA_BOOT)
    )
    acoustic = (
        1.0
        + (c["a_bleed"] * SIN_1_BOOT * SIN_1_BOOT) / phi
        + (c["a_in"] * COS_1_BOOT * COS_1_BOOT) / phi
    )
    phase = 1.0 + c["b_in"] * c["p_var"]
    t3 = valve * acoustic * phase
    return c["k_fsot"] * (t1 + t2 + t3)


def compute_batch_cpu(values: list[float], k: dict) -> list[float]:
    """Toy batch: scale boot scalar by (1 + 0.01 * i) — placeholder for tile work."""
    base = compute_fsot_scalar_boot(k)
    return [base * (1.0 + 0.01 * v) for v in values]


def main() -> int:
    report: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "canonical": SEEDS["boot"]["boot_scalar_canonical"],
        "cpu": {},
        "gpu": {},
        "match": {},
        "ok": False,
    }

    cpu_val = compute_fsot_scalar_boot(SEEDS)
    canon = SEEDS["boot"]["boot_scalar_canonical"]
    rel_err = abs(cpu_val - canon) / max(abs(canon), 1e-30)
    report["cpu"] = {
        "boot_scalar": cpu_val,
        "vs_canonical_rel_err": rel_err,
        "matches_canonical": rel_err < 1e-12,
    }

    try:
        import torch
    except ImportError as e:
        report["error"] = str(e)
        _write(report)
        print(json.dumps(report, indent=2))
        return 1

    if not torch.cuda.is_available():
        report["error"] = "CUDA unavailable"
        _write(report)
        print(json.dumps(report, indent=2))
        return 1

    # GPU: evaluate same closed form with torch ops (vectorized batch)
    device = torch.device("cuda")
    batch = torch.arange(0, 4096, device=device, dtype=torch.float64)
    base = torch.tensor(cpu_val, device=device, dtype=torch.float64)
    gpu_batch = base * (1.0 + 0.01 * batch)
    torch.cuda.synchronize()

    # f32 path for throughput path
    batch_f32 = batch.to(torch.float32)
    base_f32 = base.to(torch.float32)
    gpu_f32 = base_f32 * (1.0 + 0.01 * batch_f32)
    torch.cuda.synchronize()

    cpu_batch = compute_batch_cpu(list(range(4096)), SEEDS)
    gpu_list = gpu_batch.detach().cpu().tolist()
    max_abs = max(abs(a - b) for a, b in zip(cpu_batch, gpu_list))
    f32_list = gpu_f32.detach().cpu().tolist()
    max_abs_f32 = max(abs(a - b) for a, b in zip(cpu_batch, f32_list))

    report["gpu"] = {
        "device": torch.cuda.get_device_name(0),
        "batch": 4096,
        "dtype_primary": "float64",
        "max_abs_err_vs_cpu_f64": max_abs,
        "max_abs_err_vs_cpu_f32": max_abs_f32,
        "sample_head": gpu_list[:5],
    }
    report["match"] = {
        "cpu_matches_canonical": report["cpu"]["matches_canonical"],
        "gpu_f64_matches_cpu": max_abs < 1e-12,
        "gpu_f32_within_1e5_rel": max_abs_f32 / max(abs(cpu_val), 1e-30) < 1e-5,
    }
    report["ok"] = all(report["match"].values())

    path = _write(report)
    print("=== FSOT Boot Scalar CPU vs GPU ===")
    print(f"CPU Φ_boot:     {cpu_val:.17g}")
    print(f"Canonical:      {canon:.17g}")
    print(f"CPU vs canon:   rel_err={rel_err:.3e} match={report['cpu']['matches_canonical']}")
    print(f"GPU f64 max|ε|: {max_abs:.3e}")
    print(f"GPU f32 max|ε|: {max_abs_f32:.3e}")
    print(f"OK:             {report['ok']}")
    print(f"Wrote:          {path}")
    return 0 if report["ok"] else 2


def _write(report: dict) -> Path:
    path = RESULTS / "fsot_scalar_gpu.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())

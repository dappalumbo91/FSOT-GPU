#!/usr/bin/env python3
"""
FlashAttention-class track: FSOT sparse CUDA vs torch SDPA backends.

Standing goal (docs/GOALS.md): systematically beat FlashAttention-class
stacks on this hardware under FSOT-correct structure.

Uses torch.nn.functional.scaled_dot_product_attention with explicit backends
when available (FLASH_ATTENTION, EFFICIENT_ATTENTION, MATH).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "competitive"
OUT.mkdir(parents=True, exist_ok=True)
EXE = ROOT / "phase2_native_gpu" / "cuda" / "fsot_beat_cuda.exe"


def bench_sdpa_backend(H, S, D, backend_name, iters=80, warmup=15):
    q = torch.randn(H, S, D, device="cuda", dtype=torch.float16)
    k = torch.randn(H, S, D, device="cuda", dtype=torch.float16)
    v = torch.randn(H, S, D, device="cuda", dtype=torch.float16)
    # map name
    enable = {
        "flash": dict(enable_flash=True, enable_math=False, enable_mem_efficient=False),
        "mem_efficient": dict(enable_flash=False, enable_math=False, enable_mem_efficient=True),
        "math": dict(enable_flash=False, enable_math=True, enable_mem_efficient=False),
        "default": dict(enable_flash=True, enable_math=True, enable_mem_efficient=True),
    }[backend_name]

    def run():
        with torch.backends.cuda.sdp_kernel(**enable):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    try:
        for _ in range(warmup):
            _ = run()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            _ = run()
        torch.cuda.synchronize()
        return 1000.0 * (time.perf_counter() - t0) / iters, None
    except Exception as e:
        return None, str(e)


def parse_fsot_exe():
    if not EXE.is_file():
        return []
    r = subprocess.run([str(EXE)], capture_output=True, text=True, timeout=300)
    rows = []
    import re

    for line in (r.stdout or "").splitlines():
        m = re.search(
            r"RESULT H=(\d+) S=(\d+) D=(\d+) A_frac=([0-9.eE+-]+) "
            r"fsot_ms=([0-9.eE+-]+)",
            line,
        )
        if m:
            rows.append(
                {
                    "H": int(m.group(1)),
                    "S": int(m.group(2)),
                    "D": int(m.group(3)),
                    "A_frac": float(m.group(4)),
                    "fsot_ms": float(m.group(5)),
                }
            )
    return rows


def main():
    if not torch.cuda.is_available():
        print("CUDA required")
        return 1
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "suite": "flash_attention_track",
        "device": torch.cuda.get_device_name(0),
        "goal": "FSOT sparse CUDA always faster than FlashAttention-class SDPA under FSOT loads",
        "rows": [],
        "ok": False,
    }
    fsot_rows = parse_fsot_exe()
    backends = ["default", "flash", "mem_efficient", "math"]
    shapes = [(8, 128, 64), (8, 256, 64), (8, 512, 64), (8, 1024, 64), (9, 256, 64)]

    for H, S, D in shapes:
        fsot_ms = None
        for fr in fsot_rows:
            if fr["H"] == H and fr["S"] == S and fr["D"] == D:
                fsot_ms = fr["fsot_ms"]
                a_frac = fr["A_frac"]
                break
        if fsot_ms is None:
            # approximate: run beat exe may not have exact; skip or estimate from nearest
            a_frac = None

        row = {"H": H, "S": S, "D": D, "fsot_ms": fsot_ms, "A_frac": a_frac, "backends": {}}
        for b in backends:
            ms, err = bench_sdpa_backend(H, S, D, b)
            row["backends"][b] = {"ms": ms, "error": err}
            if ms and fsot_ms:
                row["backends"][b]["fsot_speedup"] = ms / fsot_ms
                row["backends"][b]["fsot_wins"] = fsot_ms < ms * 0.95
        report["rows"].append(row)
        print(f"H={H} S={S} D={D} fsot={fsot_ms} backends={ {k:v.get('ms') for k,v in row['backends'].items()} }")

    # score
    wins = 0
    total = 0
    for row in report["rows"]:
        for b, v in row["backends"].items():
            if v.get("ms") is not None and row.get("fsot_ms"):
                total += 1
                if v.get("fsot_wins"):
                    wins += 1
    report["summary"] = {
        "comparisons": total,
        "fsot_wins": wins,
        "win_rate": wins / max(total, 1),
        "standing_goal": "win_rate → 1.0 on all FlashAttention-class backends + larger S",
    }
    report["ok"] = total > 0
    path = OUT / "flash_attention_track.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("win_rate", report["summary"]["win_rate"], "wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

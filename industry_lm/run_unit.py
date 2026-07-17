#!/usr/bin/env python3
"""
Full industry LM unit for the Formal-GPU lab:
  export portable schema → HF baseline → FSOT bridge → longer competitive benches
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LM = Path(__file__).resolve().parent
sys.path.insert(0, str(LM))
sys.path.insert(0, str(ROOT))

from export_portable import export
from baseline_hf import run_baseline
from fsot_bridge import run_bridge

OUT = ROOT / "results" / "industry_lm"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> int:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "unit": "industry_lm_SmolLM2-135M-Instruct",
        "steps": {},
        "ok": False,
    }

    print("=== 1) Portable export ===")
    schema = export()
    report["steps"]["export"] = {"ok": schema.is_file(), "path": str(schema)}

    print("=== 2) Industry HF baseline ===")
    base = run_baseline()
    report["steps"]["baseline_hf"] = {
        "ok": True,
        "prefill_ms": base["latency"]["prefill_20x_ms"],
        "params": base["params"],
        "vram_mib": base.get("vram_allocated_mib"),
        "sample": base["generations"][0]["output"][:120] if base["generations"] else None,
    }

    print("=== 3) FSOT bridge (same weights, replaceable ops) ===")
    bridge = run_bridge()
    report["steps"]["fsot_bridge"] = {
        "ok": bridge.get("prefill_logits_ok", False),
        "attn_speedup_torch": bridge["attn_bench"].get("fsot_vs_sdpa_speedup"),
        "norm_speedup": bridge["norm_bench"].get("speedup_coh_vs_rms"),
        "sdpa_ms": bridge["attn_bench"]["industry_sdpa"]["ms_per_iter"],
        "fsot_attn_ms": bridge["attn_bench"]["fsot_sparse_torch"]["ms_per_iter"],
    }

    print("=== 4) Longer-seq + norm competitive scripts ===")
    long_py = ROOT / "competitive" / "long_seq_and_norm.py"
    if long_py.is_file():
        r = subprocess.run(
            [sys.executable, str(long_py)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        report["steps"]["long_seq_norm"] = {
            "ok": r.returncode == 0,
            "stdout_tail": (r.stdout or "")[-800:],
            "stderr_tail": (r.stderr or "")[-400:],
        }
    else:
        report["steps"]["long_seq_norm"] = {"ok": False, "reason": "script missing"}

    report["ok"] = (
        report["steps"]["export"]["ok"]
        and report["steps"]["baseline_hf"]["ok"]
        and report["steps"]["fsot_bridge"]["ok"]
    )
    out = OUT / "unit_run.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== UNIT SUMMARY ===")
    print(json.dumps(report, indent=2)[:2500])
    print("wrote", out)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

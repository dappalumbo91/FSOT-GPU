#!/usr/bin/env python3
"""
Wire THIS GPU lab to labs you already built — no from-scratch industry stall.

Sources of truth on Desktop / Archive:
  - fsot 2.1 llm  (FSOT-2.1-Instruct-0.5B, adapters, superposed generate)
  - Fsot trinary  (QEMU kernel, consensus attention, safetensors brain)
  - I:\\FSOT-Physical-Archive (403-domain verified compute + formal)
  - this lab's fsot_gpu_engine (GPU port of kernel forward)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "phase3"
RESULTS.mkdir(parents=True, exist_ok=True)

LABS = {
    "gpu_experiment": ROOT,
    "fsot_2_1_llm": Path(r"C:\Users\damia\Desktop\fsot 2.1 llm"),
    "fsot_trinary_os": Path(r"C:\Users\damia\Desktop\Fsot trinary\fsot_os"),
    "cube_trinary": Path(r"C:\Users\damia\Desktop\FSOT, Cube Block Trinary Design"),
    "archive": Path(r"I:\FSOT-Physical-Archive"),
    "fsot_compute": Path(
        r"I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full\vendor\fsot_compute.py"
    ),
    "release_05b": Path(
        r"C:\Users\damia\Desktop\fsot 2.1 llm\llm\models\release\FSOT-2.1-Instruct-0.5B"
    ),
    "trinary_forward": Path(
        r"C:\Users\damia\Desktop\Fsot trinary\fsot_os\kernel\src\forward.rs"
    ),
    "qemu_golden": Path(
        r"I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full\verification\qemu"
    ),
}


def main() -> int:
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "labs": {},
        "wiring": {
            "scalar_authority": str(LABS["fsot_compute"]),
            "gpu_engine": str(
                ROOT / "phase2_native_gpu" / "python" / "fsot_gpu_engine.py"
            ),
            "formal": {
                "lean": str(ROOT / "phase1_formal_gpu" / "lean"),
                "coq": str(ROOT / "phase1_formal_gpu" / "coq"),
                "isabelle": str(ROOT / "phase1_formal_gpu" / "isabelle"),
                "fstar": str(ROOT / "phase1_formal_gpu" / "fstar"),
            },
            "llm_release": str(LABS["release_05b"]),
            "kernel_forward": str(LABS["trinary_forward"]),
            "next_action": (
                "Run fsot_gpu_engine.py for FSOT cortical GPU train; "
                "load release 0.5B with FSOT routing from fsot 2.1 llm; "
                "keep QEMU trinary OS as bare-metal oracle."
            ),
        },
        "ok": True,
    }
    for name, path in LABS.items():
        exists = path.exists()
        report["labs"][name] = {"path": str(path), "exists": exists}
        if not exists and name in ("fsot_compute", "gpu_experiment", "archive"):
            report["ok"] = False

    # Import and touch GPU engine + optional 0.5B presence
    sys.path.insert(0, str(ROOT / "phase2_native_gpu" / "python"))
    import torch
    from fsot_gpu_engine import (  # type: ignore
        COLLAPSE_THRESHOLD,
        compute_scalar_torch,
        FSOTCorticalGPU,
        FSOTModelConfig,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    S = float(compute_scalar_torch(observed=True, D_eff=12.0, device=device).item())
    cfg = FSOTModelConfig(d_model=32, n_heads=4, n_layers=2, vocab=64)
    m = FSOTCorticalGPU(cfg, device=device)
    tokens = torch.arange(16, device=device) % 64
    with torch.no_grad():
        logits = m(tokens)
    report["live_smoke"] = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "S_sample": S,
        "logits_shape": list(logits.shape),
        "release_05b_present": LABS["release_05b"].exists(),
        "trinary_forward_present": LABS["trinary_forward"].exists(),
    }

    out = RESULTS / "lab_bridge.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== FSOT Lab Bridge ===")
    for k, v in report["labs"].items():
        print(f"  [{'OK' if v['exists'] else 'MISSING'}] {k}")
    print(f"  S_sample={S:.6g}  logits={report['live_smoke']['logits_shape']}")
    print(f"  Wrote {out}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

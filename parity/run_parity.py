#!/usr/bin/env python3
"""
Cross-language FSOT Formal-GPU parity harness.

Python fsot_lib + golden.json  ↔  Rust  ↔  Zig  ↔  formal artifacts  ↔  native CUDA
Fail-closed ledger → results/parity/parity_ledger.json
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARITY = Path(__file__).resolve().parent
GOLDEN_PATH = PARITY / "golden.json"
OUT_DIR = ROOT / "results" / "parity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def near(a: float, b: float, eps: float) -> bool:
    return abs(a - b) <= eps * max(1.0, abs(b))


def run_cmd(args: list[str], cwd: Path | None = None, timeout: float = 120) -> tuple[int, str]:
    try:
        r = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except FileNotFoundError as e:
        return 127, str(e)
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def main() -> int:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    eps = float(golden["eps_f64"])
    ledger: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "project": "FSOT-Formal-GPU",
        "golden": str(GOLDEN_PATH),
        "layers": {},
        "overall_ok": False,
    }

    # ── Python (owned lib) ────────────────────────────────────────────
    from fsot_lib import COLLAPSE_THRESHOLD, SEEDS, compute_scalar, pack_u64, unpack_u64

    codes = list(golden["pack_codes_0_to_31_mod3"])
    word = pack_u64(codes)
    S = compute_scalar(D_eff=8.0, observed=True, delta_psi=0.7)
    py = {
        "collapse_threshold": COLLAPSE_THRESHOLD,
        "phi": SEEDS.phi,
        "gamma": SEEDS.gamma,
        "k": SEEDS.k,
        "c_eff": SEEDS.c_eff,
        "p_var": SEEDS.p_var,
        "psi_con": SEEDS.psi_con,
        "pack_u64_word": word,
        "pack_ok": unpack_u64(word) == codes,
        "scalar": S,
        "scalar_match": near(S, golden["scalar_observed_D8_dp0_7"], 1e-12),
        "threshold_match": near(COLLAPSE_THRESHOLD, golden["collapse_threshold"], eps),
        "word_match": word == golden["pack_u64_word"],
    }
    py["ok"] = all(
        [
            py["pack_ok"],
            py["scalar_match"],
            py["threshold_match"],
            py["word_match"],
            near(SEEDS.k, golden["seeds"]["k"], eps),
        ]
    )
    ledger["layers"]["python"] = py

    # ── Rust ──────────────────────────────────────────────────────────
    rust_dir = PARITY / "rust_parity"
    code, out = run_cmd(["cargo", "run", "--quiet", "--release"], cwd=rust_dir, timeout=300)
    rust_layer: dict = {"build_ok": code == 0, "raw": out.strip()[-500:]}
    if code == 0:
        # last non-empty line should be JSON
        line = [ln for ln in out.splitlines() if ln.strip().startswith("{")][-1]
        data = json.loads(line)
        rust_layer.update(data)
        rust_layer["threshold_match"] = near(
            data["collapse_threshold"], golden["collapse_threshold"], eps
        )
        rust_layer["word_match"] = data["pack_u64_word"] == golden["pack_u64_word"]
        rust_layer["ok"] = (
            data.get("pack_ok", False)
            and rust_layer["threshold_match"]
            and rust_layer["word_match"]
            and near(data["k"], golden["seeds"]["k"], eps)
        )
    else:
        rust_layer["ok"] = False
    ledger["layers"]["rust"] = rust_layer

    # ── Zig ───────────────────────────────────────────────────────────
    zig_dir = PARITY / "zig_parity"
    zig_exe = zig_dir / "parity.exe"
    # Zig 0.15: default emit name is parity.exe from parity.zig
    code, out = run_cmd(
        ["zig", "build-exe", "parity.zig", "-OReleaseFast"],
        cwd=zig_dir,
        timeout=120,
    )
    zig_layer: dict = {"build_ok": code == 0, "build_raw": out.strip()[-300:]}
    if code == 0 and zig_exe.is_file():
        code2, out2 = run_cmd([str(zig_exe)], cwd=zig_dir, timeout=30)
        zig_layer["run_ok"] = code2 == 0
        if code2 == 0:
            line = [ln for ln in out2.splitlines() if ln.strip().startswith("{")][-1]
            data = json.loads(line)
            zig_layer.update(data)
            zig_layer["threshold_match"] = near(
                data["collapse_threshold"], golden["collapse_threshold"], eps
            )
            zig_layer["word_match"] = data["pack_u64_word"] == golden["pack_u64_word"]
            zig_layer["ok"] = (
                data.get("pack_ok", False)
                and zig_layer["threshold_match"]
                and zig_layer["word_match"]
            )
        else:
            zig_layer["ok"] = False
            zig_layer["raw"] = out2
    else:
        zig_layer["ok"] = False
    ledger["layers"]["zig"] = zig_layer

    # ── Formal artifacts present + lean build optional ────────────────
    formal = {
        "lean_trinary": (ROOT / "phase1_formal_gpu" / "lean" / "Trinary.lean").is_file(),
        "lean_memory": (ROOT / "phase1_formal_gpu" / "lean" / "GpuMemory.lean").is_file(),
        "coq_vo": (ROOT / "phase1_formal_gpu" / "coq" / "Trinary.vo").is_file(),
        "fstar": (ROOT / "phase1_formal_gpu" / "fstar" / "FSOTGpuBoot.fst").is_file(),
        "isabelle": (ROOT / "phase1_formal_gpu" / "isabelle" / "Trinary.thy").is_file(),
    }
    # quick lake build
    code, out = run_cmd(["lake", "build"], cwd=ROOT / "phase1_formal_gpu" / "lean", timeout=180)
    formal["lake_build_ok"] = code == 0
    formal["ok"] = all(
        [
            formal["lean_trinary"],
            formal["coq_vo"] or formal["lean_trinary"],
            formal["fstar"],
            formal["isabelle"],
            formal["lake_build_ok"],
        ]
    )
    if code != 0:
        formal["lake_raw"] = out[-400:]
    ledger["layers"]["formal"] = formal

    # ── Native CUDA ───────────────────────────────────────────────────
    from fsot_lib.backend.native_cuda import native_pack_available, run_native_pack_smoke

    if native_pack_available():
        cuda = run_native_pack_smoke()
    else:
        cuda = {"ok": False, "reason": "binary missing"}
    ledger["layers"]["cuda_native"] = cuda

    # ── overall ───────────────────────────────────────────────────────
    need = ["python", "rust", "zig", "formal"]
    ledger["overall_ok"] = all(ledger["layers"][k].get("ok") for k in need)
    # CUDA required when binary present
    if native_pack_available():
        ledger["overall_ok"] = ledger["overall_ok"] and cuda.get("ok", False)

    ledger["portability_thesis"] = (
        "Same seeds + pack + collapse across Python/Rust/Zig/formal "
        "⇒ any language implementing the contract is an FSOT GPU host."
    )
    ledger["sota_note"] = (
        "Parity proves portability. Competitive capability claims require "
        "docs/COMPETITIVE_POSITION.md gates — not this ledger alone."
    )

    out_path = OUT_DIR / "parity_ledger.json"
    out_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

    print("=== FSOT multi-language parity ===")
    for k, v in ledger["layers"].items():
        print(f"  [{'OK' if v.get('ok') else 'FAIL'}] {k}")
    print(f"overall_ok = {ledger['overall_ok']}")
    print(f"wrote {out_path}")
    return 0 if ledger["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

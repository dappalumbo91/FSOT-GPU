#!/usr/bin/env python3
"""
Single train / verify entrypoint for FSOT LLM construction (Phase 1).

Does not invent a second training stack. It:
  1. Loads config/train_fsot_llm.yaml (or --config)
  2. Optionally runs G-VERIFY (mandatory by default)
  3. Dispatches a named lever / mode to an existing climb script
  4. Writes a checkpoint-contract stub ledger for any promote path

Usage:
  python -u industry_lm/run_train.py --dry-config
  python -u industry_lm/run_train.py --verify-only
  python -u industry_lm/run_train.py --mode standard_climb
  python -u industry_lm/run_train.py --mode digit_decollapse
  python -u industry_lm/run_train.py --mode auto_refine --cycles 1 --dry-measure

Claims split: residual law = zero free params; host weights free under FSOT guide.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CFG = ROOT / "config" / "train_fsot_llm.yaml"
OUT = ROOT / "results" / "industry_lm"
OUT.mkdir(parents=True, exist_ok=True)

MODES = {
    # Primary: restore documented pure-FSOT capability (ARC≥32.5%, agree, gen)
    "recover_capability": [sys.executable, "-u", str(HERE / "run_recover_capability.py")],
    "joint_package": [sys.executable, "-u", str(HERE / "run_joint_package_climb.py")],
    "audit_package": [sys.executable, "-u", str(HERE / "audit_host_package.py")],
    "hardware_sota": [sys.executable, "-u", str(HERE / "run_hardware_sota_climb.py")],
    "standard_climb": [sys.executable, "-u", str(HERE / "run_sota_standard_climb.py")],
    "digit_decollapse": [sys.executable, "-u", str(HERE / "run_sota_digit_decollapse.py")],
    "barrier_diagnosis": [sys.executable, "-u", str(HERE / "run_barrier_diagnosis.py")],
    "auto_refine": [sys.executable, "-u", str(HERE / "run_auto_refine_loop.py")],
    "capability_smoke": [sys.executable, "-u", str(HERE / "run_capability_smoke.py")],
    "overfit_audit": [sys.executable, "-u", str(HERE / "run_overfit_audit.py")],
    "curriculum_phase1": [sys.executable, "-u", str(HERE / "run_curriculum_phase1.py")],
}


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except Exception:
        # Minimal fallback: JSON-compatible subset only
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise SystemExit(
                f"Need PyYAML to parse {path} (or pass JSON). Install pyyaml. ({e})"
            ) from e


def git_rev() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return (r.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def run_verify() -> int:
    cmd = [sys.executable, "-u", str(HERE / "fsot21_verify.py")]
    print("[run_train] G-VERIFY:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


def write_contract_stub(cfg: dict[str, Any], mode: str, verify_ok: bool | None) -> Path:
    """Checkpoint contract skeleton — filled fully by climb scripts on promote."""
    ops = cfg.get("operators") or {}
    try:
        from checkpoint_contract import build_contract, write_contract

        meta = build_contract(
            phase=f"run_train_{mode}",
            attention_mode=str(ops.get("attention_mode") or "pure_fsot_consensus"),
            layer_policy=str(ops.get("layer_policy") or "all_layers"),
            pin_verify_pass=verify_ok,
            extra={
                "config_name": cfg.get("name"),
                "host": (cfg.get("host") or {}).get("path"),
                "notes": "Entrypoint stub — promote paths write full scores.",
            },
        )
        path = write_contract(meta, append_ledger=False)
        print(f"[run_train] wrote {path}", flush=True)
        return path
    except Exception as e:
        print(f"[run_train] contract stub fallback: {e}", flush=True)
        path = OUT / "checkpoint_contract_latest.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "fsot_llm_checkpoint_contract_v1",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "mode": mode,
                    "git_rev": git_rev(),
                    "pin_verify_pass": verify_ok,
                    "claims_split": "substrate_zero_free; host_weights_free_under_fsot",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return path


def main() -> int:
    ap = argparse.ArgumentParser(description="FSOT LLM single train entrypoint")
    ap.add_argument("--config", type=Path, default=DEFAULT_CFG)
    ap.add_argument(
        "--mode",
        choices=list(MODES.keys()) + ["none"],
        default="none",
        help="Which existing climb / train path to dispatch",
    )
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--dry-config", action="store_true", help="Print resolved config and exit")
    ap.add_argument("--skip-verify", action="store_true", help="Dangerous; disabled if config.verify.mandatory")
    ap.add_argument("--cycles", type=int, default=None, help="Pass to auto_refine")
    ap.add_argument("--dry-measure", action="store_true", help="Pass to auto_refine")
    args = ap.parse_args()

    if not args.config.is_file():
        print(f"missing config: {args.config}", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    if args.dry_config:
        print(json.dumps(cfg, indent=2, default=str))
        print("--- claims ---")
        print("residual law: ZERO free parameters (FSOT-2.1-Lean)")
        print("host weights: FREE under FSOT-guided training (this repo)")
        return 0

    verify_cfg = cfg.get("verify") or {}
    mandatory = bool(verify_cfg.get("mandatory", True))
    do_verify = args.verify_only or (
        not args.skip_verify and (mandatory or bool(verify_cfg.get("pre", True)))
    )

    if args.skip_verify and mandatory:
        print("[run_train] REFUSE --skip-verify: config.verify.mandatory=true", file=sys.stderr)
        return 3

    verify_ok: bool | None = None
    if do_verify:
        rc = run_verify()
        verify_ok = rc == 0
        write_contract_stub(cfg, args.mode, verify_ok)
        if rc != 0:
            print("[run_train] G-VERIFY failed — refuse train", file=sys.stderr)
            return rc
        if args.verify_only:
            print("[run_train] verify-only OK", flush=True)
            return 0
    else:
        write_contract_stub(cfg, args.mode, None)

    if args.mode == "none":
        print("[run_train] no --mode; verify/config done. Modes:", ", ".join(MODES))
        return 0

    cmd = list(MODES[args.mode])
    if args.mode == "auto_refine":
        if args.cycles is not None:
            cmd += ["--cycles", str(args.cycles)]
        if args.dry_measure:
            cmd.append("--dry-measure")

    print("[run_train] dispatch:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())

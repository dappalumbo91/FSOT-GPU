#!/usr/bin/env python3
"""
Checkpoint contract for FSOT LLM construction.

Substrate (must follow FSOT math): seeds, collapse θ, consensus ops, pin bind.
Host weights: ordinary free parameters under FSOT-guided training — NOT seed-derived.

Weights stay gitignored (*.pt). This module writes a tracked JSON ledger beside
results so promotes are auditable without storing tensors in git.
"""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "results" / "industry_lm"
OUT.mkdir(parents=True, exist_ok=True)
LEDGER = OUT / "checkpoint_contract_ledger.jsonl"
LATEST = OUT / "checkpoint_contract_latest.json"


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


def substrate_pin() -> dict[str, Any]:
    """FSOT math substrate — zero free parameters in residual law / seeds."""
    try:
        from fsot_lib import COLLAPSE_THRESHOLD, SEEDS

        return {
            "claims": "substrate_zero_free; host_weights_free_under_fsot",
            "seeds": {
                "pi": SEEDS.pi,
                "e": SEEDS.e,
                "phi": SEEDS.phi,
                "gamma": SEEDS.gamma,
                "catalan_G": SEEDS.g_catalan,
            },
            "collapse_theta": float(COLLAPSE_THRESHOLD),
            "c_eff": float(SEEDS.c_eff),
            "p_var": float(SEEDS.p_var),
            "k": float(SEEDS.k),
            "attention_contract": "pure_fsot_consensus_no_softmax",
            "note": (
                "LLM weight tensors are NOT required to be seed-parameterized; "
                "operators/gates/pin follow FSOT math on silicon."
            ),
        }
    except Exception as e:
        return {"error": str(e), "claims": "substrate_zero_free; host_weights_free_under_fsot"}


def build_contract(
    *,
    phase: str,
    step: int | None = None,
    attention_mode: str = "pure_fsot_consensus",
    layer_policy: str = "all_layers",
    curriculum_hash: str | None = None,
    pin_verify_pass: bool | None = None,
    scores: dict[str, Any] | None = None,
    ckpt_name: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scores = scores or {}
    meta: dict[str, Any] = {
        "schema": "fsot_llm_checkpoint_contract_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": git_rev(),
        "phase": phase,
        "step": step,
        "ckpt_name": ckpt_name,
        "pin_verify_pass": pin_verify_pass,
        "attention_mode": attention_mode,
        "layer_policy": layer_policy,
        "curriculum_hash": curriculum_hash,
        "substrate": substrate_pin(),
        "scores": {
            "agree16": scores.get("agree") or scores.get("agree16"),
            "arc_min": scores.get("arc_min"),
            "arc_e": scores.get("arc_e") or scores.get("arc_easy_hold"),
            "arc_c": scores.get("arc_c") or scores.get("arc_challenge_hold"),
            "gsm_first": scores.get("gsm_first"),
            "gsm_exact": scores.get("gsm_exact"),
            "gsm_space_digit": scores.get("gsm_space_digit")
            or scores.get("space_digit")
            or scores.get("first_digit_after_space"),
            "gen_score": scores.get("gen_score"),
            "overfit_gap": scores.get("mean_overfit_gap") or scores.get("overfit_gap"),
            "balanced": scores.get("balanced") or scores.get("balanced_score"),
        },
    }
    if extra:
        meta["extra"] = extra
    return meta


def write_contract(meta: dict[str, Any], *, append_ledger: bool = True) -> Path:
    LATEST.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    if append_ledger:
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta, default=str) + "\n")
    return LATEST


def record_promote(
    *,
    phase: str,
    step: int | None,
    cap: dict[str, Any],
    gen_score: float | None = None,
    mean_overfit_gap: float | None = None,
    pin_verify_pass: bool | None = None,
    ckpt_name: str = "pure_fsot_sota_standard_best.pt",
    digit_stats: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scores = dict(cap)
    if gen_score is not None:
        scores["gen_score"] = gen_score
    if mean_overfit_gap is not None:
        scores["mean_overfit_gap"] = mean_overfit_gap
    if digit_stats:
        scores["first_digit_after_space"] = digit_stats.get("first_digit_after_space")
        scores["space_digit"] = digit_stats.get("first_digit_after_space")
        if extra is None:
            extra = {}
        extra = {
            **extra,
            "digit_argmax_top": digit_stats.get("top_argmax"),
            "digit_argmax_frac": digit_stats.get("top_frac"),
        }
    meta = build_contract(
        phase=phase,
        step=step,
        pin_verify_pass=pin_verify_pass,
        scores=scores,
        ckpt_name=ckpt_name,
        extra=extra,
    )
    write_contract(meta)
    return meta

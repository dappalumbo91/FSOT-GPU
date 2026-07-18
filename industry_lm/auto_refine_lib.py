#!/usr/bin/env python3
"""
Shared pieces for the autonomous verification & refinement loop.

Phases: data → archive/FSOT verify → measure → train lever → re-measure
→ re-verify → diagnose on fail → next lever.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

ARCHIVE_21 = Path(r"I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full")
ARCHIVE_ROOT = Path(r"I:\FSOT-Physical-Archive")
OUT = ROOT / "results" / "auto_refine"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Archive / cross-proof verification
# ---------------------------------------------------------------------------


def archive_light_verify() -> dict[str, Any]:
    """
    Fast bind to Physical-Archive cross-proof (not full multi-hour suite).
    Replays connective spine + checks report stamp + VERIFICATION_REPORT.
    """
    from fsot21_verify import (
        check_archive_stamp,
        check_connective_spine,
        check_seed_alignment,
    )

    layers = {
        "archive_stamp": check_archive_stamp(),
        "connective_spine": check_connective_spine(),
        "seed_alignment": check_seed_alignment(),
    }
    ok = all(v.get("ok") for v in layers.values())
    return {
        "mode": "archive_light",
        "ok": ok,
        "layers": layers,
        "note": "Light bind to I:\\FSOT-Physical-Archive; full suite via archive_full_cross_proof",
    }


def archive_full_cross_proof(timeout_s: int = 7200) -> dict[str, Any]:
    """
    Optional: run archive scripts/run_cross_proof_verification.py
    (hours if Coq/Isabelle present — use sparingly / nightly).
    """
    script = ARCHIVE_21 / "scripts" / "run_cross_proof_verification.py"
    if not script.is_file():
        return {"mode": "archive_full", "ok": False, "error": f"missing {script}"}
    try:
        import os

        env = os.environ.copy()
        env["FSOT_PORTABLE"] = "1"
        proc = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(ARCHIVE_21),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        report_path = ARCHIVE_21 / "data" / "cross_proof_verification_report.json"
        overall = None
        if report_path.is_file():
            rep = json.loads(report_path.read_text(encoding="utf-8"))
            overall = rep.get("overall_ok")
        return {
            "mode": "archive_full",
            "ok": bool(overall) if overall is not None else proc.returncode == 0,
            "returncode": proc.returncode,
            "overall_ok": overall,
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"mode": "archive_full", "ok": False, "error": "timeout"}
    except Exception as e:
        return {"mode": "archive_full", "ok": False, "error": str(e)}


def full_system_verify(*, include_host: bool = True, full_archive: bool = False) -> dict[str, Any]:
    """FSOT-GPU bridge + archive light (+ optional full cross-proof)."""
    from fsot21_verify import run_verification

    gpu = run_verification(include_host=include_host, write=True)
    arch = archive_light_verify()
    out: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fsot_gpu_bridge": {"ok": gpu["ok"], "failed": gpu.get("failed_layers")},
        "archive_light": arch,
        "ok": bool(gpu["ok"] and arch["ok"]),
    }
    if full_archive:
        full = archive_full_cross_proof()
        out["archive_full"] = full
        out["ok"] = out["ok"] and bool(full.get("ok"))
    return out


# ---------------------------------------------------------------------------
# Measurement snapshot
# ---------------------------------------------------------------------------


def measure_snapshot(tok, teacher, student, device, packs) -> dict[str, Any]:
    from granular_metrics import next_token_top1
    from run_sota_standard_climb import measure_all

    cap, ov = measure_all(tok, teacher, student, device, packs)
    # digit after space
    pure = [tok.encode(d, add_special_tokens=False)[0] for d in "0123456789"]
    first_ok = 0
    argmaxes = []
    for r in packs["gsm_hold"]:
        gold = re.findall(r"-?\d+", str(r["gold"]).replace(",", ""))
        gold = gold[-1] if gold else str(r["gold"]).strip()
        _, tstr, _ = next_token_top1(tok, student, device, r["prompt"] + " ")
        argmaxes.append(tstr.strip()[:1] if tstr else "?")
        first_ok += int(
            tstr.strip()[:1] == gold[0] if tstr and tstr.strip()[:1].isdigit() else False
        )
    ac = Counter(argmaxes)
    top = ac.most_common(1)[0] if ac else ("?", 0)
    # letter collapse
    from granular_metrics import eval_arc_granular

    _, items = eval_arc_granular(tok, student, device, packs["easy_hold"], arm="snap")
    preds = [it.get("pred") for it in items if it.get("pred") in list("ABCD")]
    pc = Counter(preds)
    top_l = pc.most_common(1)[0] if pc else ("?", 0)
    return {
        "cap": cap,
        "overfit": ov.as_dict(),
        "gen_score": ov.gen_score,
        "mean_overfit_gap": ov.mean_overfit_gap,
        "space_digit": first_ok / max(len(packs["gsm_hold"]), 1),
        "digit_argmax_top": top[0],
        "digit_argmax_frac": top[1] / max(len(packs["gsm_hold"]), 1),
        "arc_easy_top_letter": top_l[0],
        "arc_easy_top_frac": top_l[1] / max(len(preds), 1) if preds else 0,
    }


def gap_scores(snap: dict) -> dict[str, float]:
    """Higher = more urgent lack (0–1-ish)."""
    cap = snap["cap"]
    return {
        "digit_collapse": snap["digit_argmax_frac"] if snap["digit_argmax_top"] == "1" else 0.3,
        "space_digit_low": max(0.0, 0.55 - float(snap["space_digit"])),
        "gsm_exact_zero": 1.0 if float(cap.get("gsm_exact") or 0) < 0.01 else 0.0,
        "arc_letter_collapse": max(0.0, float(snap["arc_easy_top_frac"]) - 0.45),
        "arc_min_low": max(0.0, 0.40 - float(cap.get("arc_min") or 0)),
        "free_first_low": max(0.0, 0.45 - float(cap.get("gsm_first") or 0)),
    }


def select_lever(snap: dict, tried: list[str]) -> str:
    g = gap_scores(snap)
    # priority menu
    order = [
        ("digit_decollapse", g["digit_collapse"] + g["space_digit_low"] + 0.5 * g["gsm_exact_zero"]),
        ("arc_letter_balance", g["arc_letter_collapse"] + 0.5 * g["arc_min_low"]),
        ("standard_climb", g["arc_min_low"] + 0.3 * g["free_first_low"]),
    ]
    order.sort(key=lambda x: -x[1])
    for name, score in order:
        if score < 0.05:
            continue
        # allow retry with suffix
        if name not in tried or tried.count(name) < 2:
            return name
    return "diagnose_only"


# ---------------------------------------------------------------------------
# Train levers (bounded steps — loop-friendly)
# ---------------------------------------------------------------------------


def run_lever(
    name: str,
    *,
    max_steps: int = 200,
) -> dict[str, Any]:
    """
    Invoke existing train scripts as subprocesses with env hints,
    or run inline short train for speed.
    """
    if name == "digit_decollapse":
        return _run_script("run_sota_digit_decollapse.py", timeout_s=3600)
    if name == "arc_letter_balance":
        return _run_script("run_sota_break_barriers.py", timeout_s=3600)
    if name == "standard_climb":
        return _run_script("run_sota_standard_climb.py", timeout_s=3600)
    if name == "diagnose_only":
        return _run_script("run_barrier_diagnosis.py", timeout_s=900)
    return {"ok": False, "error": f"unknown lever {name}"}


def _run_script(script: str, timeout_s: int = 3600) -> dict[str, Any]:
    path = HERE / script
    if not path.is_file():
        return {"ok": False, "error": f"missing {path}"}
    try:
        proc = subprocess.run(
            [sys.executable, "-u", str(path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        tail = (proc.stdout or "")[-3000:]
        improved = "IMPROVED" in tail or "eligible to push" in tail or "* PROMOTED" in tail
        return {
            "ok": proc.returncode == 0 or "NO_PUSH" in tail or "IMPROVED" in tail,
            "returncode": proc.returncode,
            "improved_signal": improved,
            "stdout_tail": tail,
            "stderr_tail": (proc.stderr or "")[-1500:],
            "script": script,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "script": script}
    except Exception as e:
        return {"ok": False, "error": str(e), "script": script}


def diagnose_failure(before: dict, after: dict, lever: str, train_result: dict) -> dict[str, Any]:
    """Why this cycle failed to improve — FSOT-aware tags."""
    tags = []
    b, a = before["cap"], after["cap"]
    if after["digit_argmax_frac"] >= 0.9 and after["digit_argmax_top"] == "1":
        tags.append("digit_still_collapsed_to_1")
    if a.get("arc_min", 0) + 1e-9 < b.get("arc_min", 0) - 0.02:
        tags.append("arc_min_regressed")
    if after["gen_score"] + 1e-9 < before["gen_score"] - 0.02:
        tags.append("gen_score_regressed_overfit_risk")
    if after["arc_easy_top_frac"] >= 0.7:
        tags.append("arc_letter_still_collapsed")
    if not train_result.get("improved_signal"):
        tags.append("train_reported_no_promote")
    if a.get("gsm_exact", 0) == 0 and b.get("gsm_exact", 0) == 0:
        tags.append("gsm_exact_still_zero")
    # FSOT structure
    tags.append("fsot_response:keep_pure_consensus_no_softmax")
    if "digit_still_collapsed_to_1" in tags:
        tags.append("fsot_fix:more_digit_row_ce_balanced_first_digit")
    if "arc_letter_still_collapsed" in tags:
        tags.append("fsot_fix:letter_only_ce_or_lora_last_block")
    if "arc_min_regressed" in tags:
        tags.append("fsot_fix:restore_best_raise_retention_lower_lr")
    return {
        "tags": tags,
        "primary": tags[0] if tags else "unknown",
        "recommended_next_lever": (
            "digit_decollapse"
            if "digit_still_collapsed_to_1" in tags
            else "arc_letter_balance"
            if "arc_letter_still_collapsed" in tags
            else "standard_climb"
        ),
        "lever_this_cycle": lever,
        "train_ok": train_result.get("ok"),
    }


def cycle_improved(before: dict, after: dict) -> tuple[bool, list[str]]:
    reasons = []
    if after["space_digit"] > before["space_digit"] + 0.03:
        reasons.append(f"space_digit {before['space_digit']:.0%}→{after['space_digit']:.0%}")
    if after["digit_argmax_frac"] < before["digit_argmax_frac"] - 0.08 and after[
        "digit_argmax_top"
    ] == "1":
        reasons.append(
            f"digit_1_frac {before['digit_argmax_frac']:.0%}→{after['digit_argmax_frac']:.0%}"
        )
    if after["cap"]["arc_min"] > before["cap"]["arc_min"] + 0.02:
        reasons.append(
            f"arc_min {before['cap']['arc_min']:.0%}→{after['cap']['arc_min']:.0%}"
        )
    if after["cap"]["gsm_exact"] > before["cap"]["gsm_exact"] + 0.01:
        reasons.append("gsm_exact_up")
    if after["gen_score"] > before["gen_score"] + 0.02:
        reasons.append(f"gen_score {before['gen_score']:.3f}→{after['gen_score']:.3f}")
    if after["arc_easy_top_frac"] < before["arc_easy_top_frac"] - 0.1:
        reasons.append(
            f"letter_collapse {before['arc_easy_top_frac']:.0%}→{after['arc_easy_top_frac']:.0%}"
        )
    # floor: must not trash ARC
    if after["cap"]["arc_min"] + 1e-9 < before["cap"]["arc_min"] - 0.02:
        return False, ["arc_min_floor_broken"]
    if after["cap"]["agree"] < 0.9:
        return False, ["agree_floor"]
    return len(reasons) > 0, reasons


def write_cycle_ledger(cycle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cycle, indent=2, default=str), encoding="utf-8")

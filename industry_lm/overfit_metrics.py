#!/usr/bin/env python3
"""
Overfit / generalization error metric — curb memorization direction.

Idea (plain language):
  Every model has *two* error stories:
    • seen-data error  (train slice — what it just practiced)
    • held-out error   (fresh slice — what it should still know)

  Overfit gap  G = error_hold − error_train
    G large  → looks good on practice, worse on fresh → overfitting
    G ~ 0    → practice and fresh move together → healthier direction
    G < 0    → rare; hold better than train (noise or easy hold)

  "Direction that does not overfit":
    Prefer updates that improve hold accuracy *and* do not widen G.
    Reject updates that lower train error while raising G (classic overfit step).

Surfaces for the rest of the system:
  - scalar overfit_gap, gen_score, overfit_flag
  - accept_update(...) decision for train loops
  - measure_overfit_bundle(...) for ARC Easy/Challenge train-vs-hold
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Core scalars
# ---------------------------------------------------------------------------


def error_from_acc(acc: float | None) -> float:
    """Error = 1 − accuracy (clamped)."""
    if acc is None:
        return 1.0
    a = float(acc)
    if a < 0.0:
        a = 0.0
    if a > 1.0:
        a = 1.0
    return 1.0 - a


def overfit_gap(acc_train: float | None, acc_hold: float | None) -> float:
    """
    G = E_hold − E_train = acc_train − acc_hold

    Positive ⇒ train better than hold (overfit signature).
    """
    return error_from_acc(acc_hold) - error_from_acc(acc_train)


@dataclass
class SplitScore:
    name: str
    acc_train: float
    acc_hold: float
    n_train: int
    n_hold: int

    @property
    def err_train(self) -> float:
        return error_from_acc(self.acc_train)

    @property
    def err_hold(self) -> float:
        return error_from_acc(self.acc_hold)

    @property
    def gap(self) -> float:
        """train_acc − hold_acc (positive = overfit signature)."""
        return float(self.acc_train) - float(self.acc_hold)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "acc_train": self.acc_train,
            "acc_hold": self.acc_hold,
            "err_train": self.err_train,
            "err_hold": self.err_hold,
            "overfit_gap": self.gap,
            "n_train": self.n_train,
            "n_hold": self.n_hold,
        }


@dataclass
class OverfitReport:
    """System-facing overfit metric pack."""

    splits: list[SplitScore]
    # Combined: mean hold error, mean gap, generalization score
    mean_hold_acc: float
    mean_train_acc: float
    mean_overfit_gap: float
    max_overfit_gap: float
    # gen_score: high is better — hold acc penalized by gap
    gen_score: float
    overfit_flag: bool
    threshold_gap: float
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_train_acc": self.mean_train_acc,
            "mean_hold_acc": self.mean_hold_acc,
            "mean_overfit_gap": self.mean_overfit_gap,
            "max_overfit_gap": self.max_overfit_gap,
            "gen_score": self.gen_score,
            "overfit_flag": self.overfit_flag,
            "threshold_gap": self.threshold_gap,
            "notes": self.notes,
            "splits": [s.as_dict() for s in self.splits],
        }


def build_overfit_report(
    splits: list[SplitScore],
    *,
    threshold_gap: float = 0.08,
    gap_penalty: float = 1.0,
) -> OverfitReport:
    """
    gen_score = mean_hold_acc − gap_penalty * max(0, mean_overfit_gap)

    Steers the system toward *held-out* quality while punishing train≫hold.
    """
    if not splits:
        return OverfitReport(
            splits=[],
            mean_hold_acc=0.0,
            mean_train_acc=0.0,
            mean_overfit_gap=0.0,
            max_overfit_gap=0.0,
            gen_score=0.0,
            overfit_flag=True,
            threshold_gap=threshold_gap,
            notes="empty_splits",
        )
    mean_h = sum(s.acc_hold for s in splits) / len(splits)
    mean_t = sum(s.acc_train for s in splits) / len(splits)
    gaps = [s.gap for s in splits]
    mean_g = sum(gaps) / len(gaps)
    max_g = max(gaps)
    gen = mean_h - gap_penalty * max(0.0, mean_g)
    flag = mean_g > threshold_gap or max_g > threshold_gap * 1.5
    return OverfitReport(
        splits=splits,
        mean_hold_acc=mean_h,
        mean_train_acc=mean_t,
        mean_overfit_gap=mean_g,
        max_overfit_gap=max_g,
        gen_score=gen,
        overfit_flag=flag,
        threshold_gap=threshold_gap,
        notes="overfit_signature" if flag else "gap_within_threshold",
    )


def accept_update(
    *,
    before: OverfitReport,
    after: OverfitReport,
    min_hold_delta: float = 0.0,
    max_gap_widen: float = 0.02,
    require_gen_improve: bool = True,
) -> tuple[bool, list[str]]:
    """
    Accept a training step/checkpoint only if it is a non-overfitting direction.

    Rules (all must hold unless noted):
      1. mean_hold_acc does not fall by more than ~noise (strict: >= before + min_hold_delta)
      2. mean_overfit_gap does not widen by more than max_gap_widen
      3. if require_gen_improve: gen_score must rise
      4. never accept if after.overfit_flag and gap got worse
    """
    reasons: list[str] = []
    if after.mean_hold_acc + 1e-12 < before.mean_hold_acc + min_hold_delta:
        reasons.append(
            f"hold_acc_regressed {before.mean_hold_acc:.1%}→{after.mean_hold_acc:.1%}"
        )
    gap_delta = after.mean_overfit_gap - before.mean_overfit_gap
    if gap_delta > max_gap_widen:
        reasons.append(
            f"overfit_gap_widened {before.mean_overfit_gap:.1%}→{after.mean_overfit_gap:.1%} "
            f"(Δ{gap_delta:+.1%})"
        )
    if require_gen_improve and after.gen_score + 1e-12 < before.gen_score:
        reasons.append(
            f"gen_score_down {before.gen_score:.3f}→{after.gen_score:.3f}"
        )
    if after.overfit_flag and after.mean_overfit_gap > before.mean_overfit_gap + 1e-12:
        reasons.append("overfit_flag_and_gap_worse")
    ok = len(reasons) == 0
    if ok:
        reasons.append("non_overfit_direction")
    return ok, reasons


def direction_label(before: OverfitReport, after: OverfitReport) -> str:
    """Human-readable step diagnosis for miss/train logs."""
    d_hold = after.mean_hold_acc - before.mean_hold_acc
    d_gap = after.mean_overfit_gap - before.mean_overfit_gap
    d_train = after.mean_train_acc - before.mean_train_acc
    if d_train > 0.01 and d_hold <= 0 and d_gap > 0.01:
        return "OVERFIT_STEP"  # practice up, fresh flat/down, gap up
    if d_hold > 0 and d_gap <= 0.01:
        return "GENERALIZE_STEP"  # fresh up without gap blow-up
    if d_hold < -0.01 and d_gap > 0.01:
        return "MEMORIZE_COLLAPSE"  # worst
    if d_hold < -0.01:
        return "HOLD_REGRESS"
    if abs(d_hold) <= 0.01 and abs(d_gap) <= 0.01:
        return "FLAT"
    return "MIXED"


# ---------------------------------------------------------------------------
# Measure helpers for this lab (ARC Easy / Challenge train vs hold)
# ---------------------------------------------------------------------------


def split_disjoint(
    rows: list[dict],
    *,
    train_n: int,
    hold_n: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Disjoint shuffle split — same recipe as data-driven holds."""
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    hold = [rows[i] for i in idx[:hold_n]]
    train = [rows[i] for i in idx[hold_n : hold_n + train_n]]
    return train, hold


def score_arc_acc(eval_fn: Callable, rows: list[dict]) -> float:
    """eval_fn(rows) -> summary dict with 'exact' or float acc."""
    if not rows:
        return 0.0
    out = eval_fn(rows)
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        return float(out.get("exact") or 0.0)
    return float(out)


def measure_arc_overfit(
    eval_arc_fn: Callable,
    *,
    easy_train: list[dict],
    easy_hold: list[dict],
    challenge_train: list[dict],
    challenge_hold: list[dict],
    train_eval_n: int = 40,
    threshold_gap: float = 0.08,
) -> OverfitReport:
    """
    eval_arc_fn(rows) -> (summary_dict, items) or summary_dict with 'exact'.
    Uses first train_eval_n of each *train* pool as the 'seen' probe
    (disjoint from hold by construction of split_disjoint).
    """
    et = easy_train[:train_eval_n]
    ct = challenge_train[:train_eval_n]
    splits = [
        SplitScore(
            name="arc_easy",
            acc_train=score_arc_acc(eval_arc_fn, et),
            acc_hold=score_arc_acc(eval_arc_fn, easy_hold),
            n_train=len(et),
            n_hold=len(easy_hold),
        ),
        SplitScore(
            name="arc_challenge",
            acc_train=score_arc_acc(eval_arc_fn, ct),
            acc_hold=score_arc_acc(eval_arc_fn, challenge_hold),
            n_train=len(ct),
            n_hold=len(challenge_hold),
        ),
    ]
    return build_overfit_report(splits, threshold_gap=threshold_gap)


def measure_gsm_overfit(
    eval_gsm_fn: Callable,
    *,
    train_rows: list[dict],
    hold_rows: list[dict],
    train_eval_n: int = 40,
    metric_key: str = "first_digit",
    threshold_gap: float = 0.10,
) -> OverfitReport:
    """
    GSM free exact is often 0/0; use first_digit or tf_token_acc as the axis.
    eval_gsm_fn(rows) -> summary with metric_key.
    """

    def acc(rows):
        if not rows:
            return 0.0
        out = eval_gsm_fn(rows)
        if isinstance(out, tuple):
            out = out[0]
        return float(out.get(metric_key) or 0.0)

    tr = train_rows[:train_eval_n]
    splits = [
        SplitScore(
            name=f"gsm_{metric_key}",
            acc_train=acc(tr),
            acc_hold=acc(hold_rows),
            n_train=len(tr),
            n_hold=len(hold_rows),
        )
    ]
    return build_overfit_report(splits, threshold_gap=threshold_gap)


def combine_reports(
    *reports: OverfitReport,
    threshold_gap: float = 0.08,
) -> OverfitReport:
    """Merge multi-task overfit reports into one system metric."""
    splits: list[SplitScore] = []
    for r in reports:
        splits.extend(r.splits)
    return build_overfit_report(splits, threshold_gap=threshold_gap)


def write_overfit_ledger(
    report: OverfitReport,
    out_dir: Path,
    *,
    name: str = "overfit_metrics",
    meta: dict | None = None,
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meta": meta or {},
        **report.as_dict(),
    }
    jp = out_dir / f"{name}.json"
    mp = out_dir / f"{name}.md"
    jp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Overfit metrics — {name}",
        "",
        f"**gen_score:** {report.gen_score:.3f}  ",
        f"**mean hold acc:** {report.mean_hold_acc:.1%}  ",
        f"**mean train acc:** {report.mean_train_acc:.1%}  ",
        f"**mean overfit gap (train−hold):** {report.mean_overfit_gap:+.1%}  ",
        f"**max gap:** {report.max_overfit_gap:+.1%}  ",
        f"**overfit_flag:** {report.overfit_flag} (threshold {report.threshold_gap:.0%})  ",
        f"**note:** {report.notes}",
        "",
        "| Split | Train acc | Hold acc | Gap (train−hold) |",
        "|-------|-----------|----------|------------------|",
    ]
    for s in report.splits:
        lines.append(
            f"| {s.name} | {s.acc_train:.1%} | {s.acc_hold:.1%} | {s.gap:+.1%} |"
        )
    lines.extend(
        [
            "",
            "## How to read this",
            "",
            "- **Gap ↑ while train ↑, hold flat/↓** → overfitting direction — reject step.",
            "- **Hold ↑ and gap flat/↓** → generalization direction — accept.",
            "- **gen_score** is what the system optimizes: hold quality minus gap penalty.",
            "",
        ]
    )
    mp.write_text("\n".join(lines), encoding="utf-8")
    return {"json": str(jp), "md": str(mp)}

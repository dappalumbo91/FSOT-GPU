#!/usr/bin/env python3
"""Re-measure base vs promoted ckpt multiple times; only confirm IMPROVED if avg beats."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from fsot_layer_swap import swap_all_layers  # noqa: E402
from real_data_packs import load_gsm8k_test  # noqa: E402
from run_data_driven_push import (  # noqa: E402
    CKPT,
    DATA,
    load_model,
    measure,
    slim,
    split_arc,
)


def eval_ck(path: Path, teacher, device, gsm_eval, hold_e, hold_c, reps: int = 3):
    tok, m = load_model(device)
    swap_all_layers(m)
    ck = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(ck["state_dict"], strict=False)
    rows = []
    for i in range(reps):
        met = slim(measure(tok, teacher, m, device, gsm_eval, hold_e, hold_c))
        rows.append(met)
        print(
            f"  {path.name} rep{i}: min={met['arc_min']:.1%} "
            f"E={met['arc_e']:.1%} C={met['arc_c']:.1%} "
            f"first={met['gsm_first']:.1%} tf={met['gsm_tf']:.1%} bal={met['balanced']:.3f}"
        )
    del m
    if device == "cuda":
        torch.cuda.empty_cache()
    return rows


def avg(rows, k):
    vals = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    return sum(vals) / max(len(vals), 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_t, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    gsm_eval = load_gsm8k_test(40)
    for r in gsm_eval:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"
    _, hold_e = split_arc(DATA / "ARC-Easy_train.csv", 2500, 60, 17)
    _, hold_c = split_arc(DATA / "ARC-Challenge_train.csv", 1500, 40, 19)

    base_path = CKPT / "pure_fsot_answer_locked_best.pt"
    # Prefer original answer_locked before granular overwrite — if data_driven
    # overwrote granular, still compare to answer_locked host used as start.
    # Also try 12x3 for reference.
    print("=== BASE answer_locked ===")
    b = eval_ck(base_path, teacher, device, gsm_eval, hold_e, hold_c, 3)

    prom_path = CKPT / "pure_fsot_data_driven_best.pt"
    d = None
    if prom_path.is_file():
        print("=== PROMOTED data_driven ===")
        d = eval_ck(prom_path, teacher, device, gsm_eval, hold_e, hold_c, 3)

    num_keys = [
        k
        for k in b[0]
        if isinstance(b[0][k], (int, float))
    ]
    report = {
        "base_avg": {k: avg(b, k) for k in num_keys},
        "base_reps": b,
    }
    if d:
        report["prom_avg"] = {k: avg(d, k) for k in num_keys}
        report["prom_reps"] = d
        report["delta_arc_min"] = avg(d, "arc_min") - avg(b, "arc_min")
        report["delta_balanced"] = avg(d, "balanced") - avg(b, "balanced")
        report["delta_gsm_first"] = avg(d, "gsm_first") - avg(b, "gsm_first")
        # Confirm improve if avg arc_min up by >=1pt OR balanced up >=0.03 with arc_min not down
        conf = (
            report["delta_arc_min"] >= 0.01
            or (
                report["delta_balanced"] >= 0.03
                and report["delta_arc_min"] >= -0.005
            )
            or report["delta_gsm_first"] >= 0.05
        )
        report["confirmed_improve"] = bool(conf)
    else:
        report["confirmed_improve"] = False

    out = ROOT / "results" / "industry_lm" / "gate_verify.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print("=== VERDICT ===")
    if d:
        print(
            f"base min={avg(b,'arc_min'):.1%} prom min={avg(d,'arc_min'):.1%} "
            f"Δmin={report['delta_arc_min']:+.1%}"
        )
        print(
            f"base bal={avg(b,'balanced'):.3f} prom bal={avg(d,'balanced'):.3f} "
            f"Δbal={report['delta_balanced']:+.3f}"
        )
        print(
            "CONFIRMED_IMPROVE"
            if report["confirmed_improve"]
            else "NOT_CONFIRMED — do not push"
        )
    else:
        print("no promoted ckpt")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

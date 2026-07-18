#!/usr/bin/env python3
"""
Audit pure-FSOT hosts with the overfit gap metric.

Surfaces train-vs-hold error so the rest of the system can reject
overfitting directions (see overfit_metrics.accept_update).
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from fsot_layer_swap import swap_all_layers  # noqa: E402
from granular_metrics import eval_arc_granular, eval_gsm_granular  # noqa: E402
from overfit_metrics import (  # noqa: E402
    combine_reports,
    measure_arc_overfit,
    measure_gsm_overfit,
    split_disjoint,
    write_overfit_ledger,
)
from real_data_packs import load_arc_train, load_gsm8k_test, load_gsm8k_train  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
CKPT = OUT / "checkpoints"
DATA = Path(r"D:\training data")


def load_host(device, ckpt: Path | None):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    swap_all_layers(m)
    meta = {}
    if ckpt and ckpt.is_file():
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        m.load_state_dict(ck["state_dict"], strict=False)
        meta = {
            k: ck.get(k)
            for k in (
                "phase",
                "step",
                "arc_min",
                "arc_easy_hold",
                "arc_challenge_hold",
                "gsm_first",
                "agree16",
            )
        }
        print("loaded", ckpt.name, meta)
    m.eval()
    return tok, m, meta


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== OVERFIT AUDIT ===")

    # Disjoint splits (fixed seeds — comparable over time)
    easy_all = load_arc_train(DATA / "ARC-Easy_train.csv", None)
    ch_all = load_arc_train(DATA / "ARC-Challenge_train.csv", None)
    easy_tr, easy_h = split_disjoint(easy_all, train_n=2500, hold_n=60, seed=17)
    ch_tr, ch_h = split_disjoint(ch_all, train_n=1500, hold_n=40, seed=19)
    gsm_hold = load_gsm8k_test(40)
    for r in gsm_hold:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"
    gsm_tr = load_gsm8k_train(200)
    # short prompts for train probe
    gsm_tr_probe = []
    for r in gsm_tr:
        q = r["text"].split("\n")[0]
        if not q.startswith("Question:"):
            q = "Question: " + q
        gsm_tr_probe.append({"prompt": f"{q}\n####", "gold": r["gold"]})

    hosts = [
        ("baseline_hf", None, False),
        ("data_driven_best", CKPT / "pure_fsot_data_driven_best.pt", True),
        ("12x3_best", CKPT / "pure_fsot_12x3_best.pt", True),
        ("granular_best", CKPT / "pure_fsot_granular_best.pt", True),
    ]

    board = []
    for label, path, do_swap in hosts:
        if path is not None and not path.is_file():
            print("skip missing", label)
            continue
        print(f"\n--- {label} ---")
        if not do_swap:
            tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            m = AutoModelForCausalLM.from_pretrained(
                str(MODEL), dtype=torch.float32, trust_remote_code=True
            ).to(device)
            m.eval()
            meta = {"phase": "hf_baseline"}
        else:
            tok, m, meta = load_host(device, path)

        def eval_arc(rows, _m=m, _tok=tok):
            return eval_arc_granular(_tok, _m, device, rows, arm=label)

        def eval_gsm(rows, _m=m, _tok=tok):
            return eval_gsm_granular(_tok, _m, device, rows, arm=label)

        arc_rep = measure_arc_overfit(
            eval_arc,
            easy_train=easy_tr,
            easy_hold=easy_h,
            challenge_train=ch_tr,
            challenge_hold=ch_h,
            train_eval_n=40,
            threshold_gap=0.08,
        )
        gsm_rep = measure_gsm_overfit(
            eval_gsm,
            train_rows=gsm_tr_probe,
            hold_rows=gsm_hold,
            train_eval_n=40,
            metric_key="first_digit",
            threshold_gap=0.10,
        )
        comb = combine_reports(arc_rep, gsm_rep, threshold_gap=0.08)
        paths = write_overfit_ledger(
            comb,
            OUT,
            name=f"overfit_{label}",
            meta={"host": label, "ckpt": str(path) if path else None, **meta},
        )
        print(
            f"  train={comb.mean_train_acc:.0%} hold={comb.mean_hold_acc:.0%} "
            f"gap={comb.mean_overfit_gap:+.0%} gen={comb.gen_score:.3f} "
            f"flag={comb.overfit_flag}"
        )
        for s in comb.splits:
            print(
                f"    {s.name}: train={s.acc_train:.0%} hold={s.acc_hold:.0%} gap={s.gap:+.0%}"
            )
        board.append(
            {
                "host": label,
                "ckpt": str(path) if path else None,
                "meta": meta,
                "report": comb.as_dict(),
                "paths": paths,
            }
        )
        del m
        if device == "cuda":
            torch.cuda.empty_cache()

    # rank by gen_score (higher = better generalization direction)
    ranked = sorted(board, key=lambda b: b["report"]["gen_score"], reverse=True)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "overfit_gap = train_acc − hold_acc; gen_score = hold − penalty*gap",
        "how_to_use": {
            "reject_step_if": "hold drops OR overfit_gap widens >2pts OR gen_score falls",
            "accept_step_if": "overfit_metrics.accept_update(before, after) is True",
            "system_surface": [
                "gen_score",
                "mean_overfit_gap",
                "overfit_flag",
                "direction_label(before, after)",
            ],
        },
        "ranked_by_gen_score": [
            {
                "host": b["host"],
                "gen_score": b["report"]["gen_score"],
                "mean_hold_acc": b["report"]["mean_hold_acc"],
                "mean_overfit_gap": b["report"]["mean_overfit_gap"],
                "overfit_flag": b["report"]["overfit_flag"],
            }
            for b in ranked
        ],
        "hosts": board,
    }
    outp = OUT / "overfit_audit.json"
    outp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = [
        "# Overfit audit",
        "",
        "Brother's tidbit, made operational: **error that shows over to the system**",
        "so we can prefer non-overfitting directions.",
        "",
        "## Metric",
        "",
        "| Symbol | Meaning |",
        "|--------|---------|",
        "| `err = 1 − acc` | Error on a split |",
        "| `overfit_gap = acc_train − acc_hold` | Positive ⇒ practice ≫ fresh |",
        "| `gen_score = mean_hold − penalty·max(0, gap)` | What we want to **maximize** |",
        "| `overfit_flag` | Gap above threshold |",
        "",
        "## Host ranking (by gen_score)",
        "",
        "| Host | Hold acc | Overfit gap | gen_score | Flag |",
        "|------|----------|-------------|-----------|------|",
    ]
    for b in ranked:
        r = b["report"]
        md.append(
            f"| {b['host']} | {r['mean_hold_acc']:.0%} | {r['mean_overfit_gap']:+.0%} | "
            f"**{r['gen_score']:.3f}** | {r['overfit_flag']} |"
        )
    md.extend(
        [
            "",
            "## Use in training",
            "",
            "```python",
            "from overfit_metrics import accept_update, direction_label",
            "ok, reasons = accept_update(before=rep0, after=rep1)",
            "if not ok: restore_checkpoint()  # curb overfit direction",
            "```",
            "",
            f"Ledger: `{outp.name}`",
            "",
        ]
    )
    (OUT / "OVERFIT_AUDIT.md").write_text("\n".join(md), encoding="utf-8")
    print("\n=== RANK gen_score ===")
    for b in ranked:
        r = b["report"]
        print(
            f"  {b['host']}: gen={r['gen_score']:.3f} hold={r['mean_hold_acc']:.0%} "
            f"gap={r['mean_overfit_gap']:+.0%} flag={r['overfit_flag']}"
        )
    print("wrote", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

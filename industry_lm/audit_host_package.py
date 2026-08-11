#!/usr/bin/env python3
"""
Joint package audit — rank pure-FSOT hosts by simultaneous capability,
not single-axis 'wins' that hide regressions.
"""
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
from overfit_metrics import split_disjoint  # noqa: E402
from real_data_packs import load_arc_train, load_gsm8k_test, load_gsm8k_train  # noqa: E402
from run_sota_digit_decollapse import digit_argmax_stats, pure_digit_ids  # noqa: E402
from run_sota_standard_climb import CKPT, DATA, load_model, measure_all  # noqa: E402

OUT = ROOT / "results" / "industry_lm"


def package_score(arc_min, agree, gen, space_dig, mode_frac, gsm_first) -> float:
    # Penalize mode collapse; do NOT let digit "win" by destroying ARC.
    return (
        2.0 * float(arc_min)
        + 1.0 * float(agree)
        + 0.5 * float(gen)
        + 0.4 * float(space_dig)
        + 0.2 * float(gsm_first)
        - 0.5 * max(0.0, float(mode_frac) - 0.35)
    )


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    easy_all = load_arc_train(DATA / "ARC-Easy_train.csv", None)
    ch_all = load_arc_train(DATA / "ARC-Challenge_train.csv", None)
    easy_tr, easy_h = split_disjoint(easy_all, train_n=2500, hold_n=60, seed=17)
    ch_tr, ch_h = split_disjoint(ch_all, train_n=1500, hold_n=40, seed=19)
    gsm_hold = load_gsm8k_test(40)
    for r in gsm_hold:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"
    gsm_probe = []
    for r in load_gsm8k_train(200):
        q = r["text"].split("\n")[0]
        if not q.startswith("Question:"):
            q = "Question: " + q
        gsm_probe.append({"prompt": f"{q}\n####", "gold": r["gold"]})
    packs = dict(
        easy_train=easy_tr,
        easy_hold=easy_h,
        ch_train=ch_tr,
        ch_hold=ch_h,
        gsm_hold=gsm_hold,
        gsm_train_probe=gsm_probe,
    )

    priority = [
        "pure_fsot_sota_standard_best.pt",
        "pure_fsot_data_driven_best.pt",
        "pure_fsot_granular_best.pt",
        "pure_fsot_arc_locked_best.pt",
        "pure_fsot_sota_climb_best.pt",
        "pure_fsot_answer_locked_best.pt",
        "pure_fsot_hardware_sota_best.pt",
        "pure_fsot_barrier_lab_best.pt",
        "pure_fsot_12x3_best.pt",
        "pure_fsot_agree100_best.pt",
        "pure_fsot_curriculum_best.pt",
        "pure_fsot_digit_lab_best.pt",
        "pure_fsot_gsm_locked_best.pt",
        "pure_fsot_realdata_best.pt",
    ]
    paths = [CKPT / n for n in priority if (CKPT / n).is_file()]

    tok_t, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    rows = []
    for src in paths:
        print("measure", src.name, flush=True)
        tok, student = load_model(device)
        swap_all_layers(student)
        ck = torch.load(src, map_location=device, weights_only=False)
        student.load_state_dict(ck["state_dict"], strict=False)
        pure = pure_digit_ids(tok)
        cap, ov = measure_all(tok, teacher, student, device, packs)
        d = digit_argmax_stats(tok, student, device, gsm_hold, pure)
        rec = {
            "name": src.name,
            "arc_min": float(cap["arc_min"]),
            "arc_e": float(cap["arc_e"]),
            "arc_c": float(cap["arc_c"]),
            "agree": float(cap["agree"]),
            "gsm_first": float(cap["gsm_first"]),
            "gsm_exact": float(cap["gsm_exact"]),
            "gen": float(ov.gen_score),
            "space_dig": float(d["first_digit_after_space"]),
            "mode_top": d["top_argmax"],
            "mode_frac": float(d["top_frac"]),
            "phase": ck.get("phase"),
        }
        rec["package"] = package_score(
            rec["arc_min"],
            rec["agree"],
            rec["gen"],
            rec["space_dig"],
            rec["mode_frac"],
            rec["gsm_first"],
        )
        rows.append(rec)
        print(
            f"  arc={rec['arc_min']:.0%} dig={rec['space_dig']:.0%} "
            f"mode={rec['mode_top']}@{rec['mode_frac']:.0%} "
            f"agree={rec['agree']:.0%} gen={rec['gen']:.3f} pkg={rec['package']:.3f}",
            flush=True,
        )
        del student
        if device == "cuda":
            torch.cuda.empty_cache()

    rows.sort(key=lambda x: -x["package"])
    print("\n=== RANK BY JOINT PACKAGE (not single-axis) ===", flush=True)
    for r in rows:
        print(
            f"{r['package']:.3f}  arc={r['arc_min']:.0%} dig={r['space_dig']:.0%} "
            f"mode={r['mode_top']}@{r['mode_frac']:.0%} agree={r['agree']:.0%} "
            f"gen={r['gen']:.3f}  {r['name']}",
            flush=True,
        )

    out = OUT / "host_package_audit.json"
    out.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    md = OUT / "HOST_PACKAGE_AUDIT.md"
    lines = [
        "# Host package audit (joint capability)",
        "",
        "Single-axis climbs that regress other floors are **not** FSOT application.",
        "Rank is **package score**: ARC + agree + gen + digit − mode-collapse penalty.",
        "",
        "| Package | ARC min | Space dig | Mode | Agree | gen | Host |",
        "|---------|---------|-----------|------|-------|-----|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['package']:.3f} | {r['arc_min']:.0%} | {r['space_dig']:.0%} | "
            f"{r['mode_top']}@{r['mode_frac']:.0%} | {r['agree']:.0%} | "
            f"{r['gen']:.3f} | `{r['name']}` |"
        )
    best = rows[0] if rows else {}
    lines.extend(
        [
            "",
            f"**Spine host (highest package):** `{best.get('name')}`",
            "",
            "Rule: all future train must **start from spine** and **reject any step**",
            "that drops any frozen floor (ARC, agree, gen, mode mass, space-digit).",
        ]
    )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out)
    print("wrote", md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

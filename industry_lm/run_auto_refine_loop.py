#!/usr/bin/env python3
"""
Autonomous verification & refinement loop for pure-FSOT industry host.

Loop:
  1. Load real data packs (D:\\training data)
  2. Verify — FSOT-GPU bridge + Physical-Archive light cross-proof
     (optional --full-archive for full cross_proof runner)
  3. Measure capability / overfit / digit collapse / letter collapse
  4. Select FSOT lever from competitive gap priority
  5. Train (bounded existing climb scripts)
  6. Re-measure + re-verify
  7. If improved → promote note in ledger
     If not → diagnose why + queue FSOT fix lever next cycle
  8. Repeat for --cycles N

Does NOT force-push to GitHub (operator reviews). Writes:
  results/auto_refine/loop_YYYYMMDD_HHMMSS/
    cycle_NN.json, LOOP_SUMMARY.md, latest_snapshot.json

Examples:
  python -u industry_lm/run_auto_refine_loop.py --cycles 1 --dry-measure
  python -u industry_lm/run_auto_refine_loop.py --cycles 3
  python -u industry_lm/run_auto_refine_loop.py --cycles 1 --full-archive
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from auto_refine_lib import (  # noqa: E402
    OUT,
    cycle_improved,
    diagnose_failure,
    full_system_verify,
    gap_scores,
    measure_snapshot,
    run_lever,
    select_lever,
    write_cycle_ledger,
)
from fsot_layer_swap import swap_all_layers  # noqa: E402
from overfit_metrics import split_disjoint  # noqa: E402
from real_data_packs import load_arc_train, load_gsm8k_test, load_gsm8k_train  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
CKPT = ROOT / "results" / "industry_lm" / "checkpoints"
DATA = Path(r"D:\training data")


def load_data_packs():
    """Grab real training/eval data."""
    easy_all = load_arc_train(DATA / "ARC-Easy_train.csv", None)
    ch_all = load_arc_train(DATA / "ARC-Challenge_train.csv", None)
    easy_tr, easy_h = split_disjoint(easy_all, train_n=2500, hold_n=60, seed=17)
    ch_tr, ch_h = split_disjoint(ch_all, train_n=1500, hold_n=40, seed=19)
    gsm_hold = load_gsm8k_test(40)
    for r in gsm_hold:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"
    gsm_probe = []
    for r in load_gsm8k_train(400):
        q = r["text"].split("\n")[0]
        if not q.startswith("Question:"):
            q = "Question: " + q
        gsm_probe.append({"prompt": f"{q}\n####", "gold": r["gold"]})
    return {
        "easy_train": easy_tr,
        "easy_hold": easy_h,
        "ch_train": ch_tr,
        "ch_hold": ch_h,
        "gsm_hold": gsm_hold,
        "gsm_train_probe": gsm_probe,
        "data_root": str(DATA),
        "n_arc_easy_train": len(easy_tr),
        "n_arc_ch_train": len(ch_tr),
        "n_gsm_hold": len(gsm_hold),
    }


def load_host(device):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    student = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    swap_all_layers(student)
    src = None
    for cand in (
        CKPT / "pure_fsot_sota_standard_best.pt",
        CKPT / "pure_fsot_data_driven_best.pt",
        CKPT / "pure_fsot_12x3_best.pt",
    ):
        if cand.is_file():
            src = cand
            break
    if src is None:
        raise FileNotFoundError("no pure_fsot_*.pt host checkpoint")
    ck = torch.load(src, map_location=device, weights_only=False)
    student.load_state_dict(ck["state_dict"], strict=False)
    teacher.eval()
    student.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return tok, teacher, student, src


def slim_snap(s: dict) -> dict:
    return {
        "arc_min": s["cap"]["arc_min"],
        "arc_e": s["cap"]["arc_e"],
        "arc_c": s["cap"]["arc_c"],
        "gsm_first": s["cap"]["gsm_first"],
        "gsm_exact": s["cap"]["gsm_exact"],
        "agree": s["cap"]["agree"],
        "gen_score": s["gen_score"],
        "space_digit": s["space_digit"],
        "digit_argmax": f"{s['digit_argmax_top']}@{s['digit_argmax_frac']:.0%}",
        "arc_letter_top": f"{s['arc_easy_top_letter']}@{s['arc_easy_top_frac']:.0%}",
    }


def main():
    ap = argparse.ArgumentParser(description="FSOT-GPU auto verify+refine loop")
    ap.add_argument("--cycles", type=int, default=2, help="number of refine cycles")
    ap.add_argument(
        "--full-archive",
        action="store_true",
        help="run full Physical-Archive cross_proof (slow)",
    )
    ap.add_argument(
        "--dry-measure",
        action="store_true",
        help="only data+verify+measure (no train)",
    )
    ap.add_argument(
        "--force-lever",
        type=str,
        default="",
        help="force lever: digit_decollapse|arc_letter_balance|standard_climb|diagnose_only",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = OUT / f"loop_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== FSOT-GPU AUTO REFINE LOOP ===")
    print(f"run_dir={run_dir}")
    print(f"cycles={args.cycles} full_archive={args.full_archive} dry={args.dry_measure}")

    # --- 1 DATA ---
    print("\n[1] DATA — load real packs")
    packs = load_data_packs()
    print(
        f"  ARC easy train={packs['n_arc_easy_train']} hold={len(packs['easy_hold'])} "
        f"ch train={packs['n_arc_ch_train']} gsm hold={packs['n_gsm_hold']}"
    )
    meta = {
        "started": datetime.now(timezone.utc).isoformat(),
        "cycles": args.cycles,
        "full_archive": args.full_archive,
        "dry_measure": args.dry_measure,
        "data": {
            k: packs[k]
            for k in (
                "data_root",
                "n_arc_easy_train",
                "n_arc_ch_train",
                "n_gsm_hold",
            )
        },
        "cycles_log": [],
        "promotions": [],
    }

    # --- 2 VERIFY PRE ---
    print("\n[2] VERIFY PRE — FSOT-GPU + archive light cross-proof")
    v0 = full_system_verify(include_host=True, full_archive=args.full_archive)
    print(f"  system_verify ok={v0['ok']} bridge={v0['fsot_gpu_bridge']['ok']} "
          f"archive_light={v0['archive_light']['ok']}")
    if not v0["ok"]:
        print("VERIFY PRE FAILED — abort loop (will not train broken theory bind)")
        meta["aborted"] = "verify_pre_failed"
        meta["verify_pre"] = v0
        (run_dir / "LOOP_SUMMARY.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        return 1

    # --- load host for measure ---
    print("\n[3] HOST load")
    tok, teacher, student, src = load_host(device)
    print(f"  host={src.name}")

    print("\n[4] MEASURE baseline")
    student.eval()
    snap0 = measure_snapshot(tok, teacher, student, device, packs)
    print("  ", slim_snap(snap0))
    print("  gaps", {k: round(v, 3) for k, v in gap_scores(snap0).items()})
    (run_dir / "baseline_snapshot.json").write_text(
        json.dumps({"host": src.name, "snap": slim_snap(snap0), "full": snap0}, indent=2, default=str),
        encoding="utf-8",
    )

    if args.dry_measure:
        meta["mode"] = "dry_measure"
        meta["baseline"] = slim_snap(snap0)
        meta["final_snapshot"] = slim_snap(snap0)
        meta["verify_pre"] = v0
        meta["n_promotions"] = 0
        meta["finished"] = datetime.now(timezone.utc).isoformat()
        (run_dir / "LOOP_SUMMARY.json").write_text(
            json.dumps(meta, indent=2, default=str), encoding="utf-8"
        )
        _write_summary_md(run_dir, meta, snap0, snap0)
        (OUT / "LATEST_LOOP.md").write_text(
            (run_dir / "LOOP_SUMMARY.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        print("dry-measure done — no train")
        return 0

    tried: list[str] = []
    best_snap = snap0
    del student
    del teacher
    if device == "cuda":
        torch.cuda.empty_cache()

    for cyc in range(1, args.cycles + 1):
        print(f"\n======== CYCLE {cyc}/{args.cycles} ========")
        t0 = time.time()
        cycle: dict = {
            "cycle": cyc,
            "started": datetime.now(timezone.utc).isoformat(),
        }

        # select lever
        lever = args.force_lever.strip() or select_lever(best_snap, tried)
        tried.append(lever)
        cycle["lever"] = lever
        print(f"[cycle {cyc}] lever={lever}")

        # VERIFY before train (light, host optional if last train left ckpt)
        print(f"[cycle {cyc}] VERIFY before train")
        v_pre = full_system_verify(include_host=True, full_archive=False)
        cycle["verify_pre"] = {
            "ok": v_pre["ok"],
            "failed_bridge": v_pre["fsot_gpu_bridge"].get("failed"),
            "archive_light_ok": v_pre["archive_light"]["ok"],
        }
        if not v_pre["ok"]:
            cycle["aborted"] = "verify_pre_failed"
            cycle["diagnose"] = {
                "tags": ["verify_broken"],
                "fsot_fix": "restore_last_good_ckpt_rerun_fsot21_verify",
            }
            write_cycle_ledger(cycle, run_dir / f"cycle_{cyc:02d}.json")
            meta["cycles_log"].append(cycle)
            print("  verify failed — skip train, diagnose only")
            train_result = run_lever("diagnose_only")
            cycle["train"] = train_result
            continue

        # TRAIN
        print(f"[cycle {cyc}] TRAIN {lever}")
        train_result = run_lever(lever)
        cycle["train"] = {
            "ok": train_result.get("ok"),
            "improved_signal": train_result.get("improved_signal"),
            "script": train_result.get("script"),
            "error": train_result.get("error"),
            "stdout_tail": train_result.get("stdout_tail", "")[-1500:],
        }
        print(
            f"  train ok={train_result.get('ok')} promote_signal={train_result.get('improved_signal')}"
        )

        # MEASURE after
        print(f"[cycle {cyc}] MEASURE after")
        tok, teacher, student, src2 = load_host(device)
        snap1 = measure_snapshot(tok, teacher, student, device, packs)
        cycle["measure_before"] = slim_snap(best_snap)
        cycle["measure_after"] = slim_snap(snap1)
        print("  after", slim_snap(snap1))

        # VERIFY after
        print(f"[cycle {cyc}] VERIFY after")
        v_post = full_system_verify(include_host=True, full_archive=args.full_archive)
        cycle["verify_post"] = {
            "ok": v_post["ok"],
            "archive_light_ok": v_post["archive_light"]["ok"],
            "archive_full": v_post.get("archive_full", {}).get("ok")
            if args.full_archive
            else None,
        }
        print(f"  verify_post ok={v_post['ok']}")

        improved, reasons = cycle_improved(best_snap, snap1)
        if not v_post["ok"]:
            improved = False
            reasons = ["verify_post_failed"] + reasons

        cycle["improved"] = improved
        cycle["improve_reasons"] = reasons
        cycle["elapsed_s"] = time.time() - t0

        if improved:
            print(f"  ✓ CYCLE IMPROVED: {reasons}")
            best_snap = snap1
            meta["promotions"].append(
                {"cycle": cyc, "lever": lever, "reasons": reasons, "snap": slim_snap(snap1)}
            )
            cycle["diagnose"] = {"tags": ["success"], "recommended_next_lever": select_lever(snap1, tried)}
        else:
            print(f"  ✗ CYCLE NO IMPROVE — diagnose")
            diag = diagnose_failure(best_snap, snap1, lever, train_result)
            cycle["diagnose"] = diag
            print(f"  primary={diag['primary']} next={diag['recommended_next_lever']}")
            print(f"  tags={diag['tags']}")
            # queue recommended lever for next cycle
            if diag["recommended_next_lever"] not in tried[-1:]:
                pass  # select_lever will pick based on gaps

        write_cycle_ledger(cycle, run_dir / f"cycle_{cyc:02d}.json")
        meta["cycles_log"].append(
            {k: cycle[k] for k in cycle if k not in ("train",) or True}
        )
        # free GPU between cycles
        del student
        del teacher
        if device == "cuda":
            torch.cuda.empty_cache()

    meta["finished"] = datetime.now(timezone.utc).isoformat()
    meta["final_snapshot"] = slim_snap(best_snap)
    meta["baseline_snapshot"] = slim_snap(snap0)
    meta["n_promotions"] = len(meta["promotions"])
    (run_dir / "LOOP_SUMMARY.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "latest_snapshot.json").write_text(
        json.dumps(slim_snap(best_snap), indent=2), encoding="utf-8"
    )
    _write_summary_md(run_dir, meta, snap0, best_snap)
    # also copy pointer at auto_refine/LATEST
    (OUT / "LATEST_LOOP.md").write_text(
        (run_dir / "LOOP_SUMMARY.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print("\n=== LOOP DONE ===")
    print(f"promotions={meta['n_promotions']} dir={run_dir}")
    print(f"baseline {slim_snap(snap0)}")
    print(f"final    {slim_snap(best_snap)}")
    return 0


def _write_summary_md(run_dir: Path, meta: dict, snap0: dict, snap_f: dict) -> None:
    lines = [
        "# Auto refine loop summary",
        "",
        f"**Started:** {meta.get('started')}  ",
        f"**Finished:** {meta.get('finished', '—')}  ",
        f"**Cycles:** {meta.get('cycles')}  ",
        f"**Promotions:** {meta.get('n_promotions', 0)}  ",
        "",
        "## Baseline → best",
        "",
        "| Axis | Baseline | Best |",
        "|------|----------|------|",
    ]
    b, a = slim_snap(snap0), slim_snap(snap_f)
    for k in b:
        lines.append(f"| {k} | {b[k]} | {a[k]} |")
    lines.extend(["", "## Cycles", ""])
    for c in meta.get("cycles_log", []):
        if not isinstance(c, dict):
            continue
        lines.append(
            f"- **Cycle {c.get('cycle')}** lever=`{c.get('lever')}` "
            f"improved={c.get('improved')} reasons={c.get('improve_reasons')} "
            f"diag={c.get('diagnose', {}).get('primary')}"
        )
    lines.extend(
        [
            "",
            "## Gaps inventory",
            "",
            "See [`docs/COMPETITIVE_GAPS.md`](../../docs/COMPETITIVE_GAPS.md).",
            "",
            "## Verification",
            "",
            "Each cycle: FSOT-GPU bridge + Physical-Archive **light** cross-proof "
            "(connective spine + stamp + seed θ). Optional `--full-archive` for full suite.",
            "",
        ]
    )
    (run_dir / "LOOP_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

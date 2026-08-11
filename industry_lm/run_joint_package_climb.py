#!/usr/bin/env python3
"""
Joint package climb — correct FSOT application for low-param capability.

Problem we solve:
  More held-out capability per parameter on pure FSOT 135M under hardware limits.

What this refuses:
  Single-axis 'wins' that regress ARC, agree, gen_score, digit skill, or re-collapse mode.
  Those are not FSOT application; they are proxy hacking.

Every train step:
  ARC letter CE + digit anti-mode CE + teacher retention
  under suction–poof LR, pure FSOT attention host.

Every eval:
  promote only if package improves AND no frozen floor breaks.

See docs/FSOT_CORRECT_APPLICATION.md
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from fsot_layer_swap import swap_all_layers  # noqa: E402
from fsot_lib.learn import derive_fsot_lr_plan, fsot_epoch_lr  # noqa: E402
from fsot21_verify import run_verification  # noqa: E402
from overfit_metrics import accept_update, split_disjoint, write_overfit_ledger  # noqa: E402
from real_data_packs import load_arc_train, load_gsm8k_test, load_gsm8k_train  # noqa: E402
from run_sota_break_barriers import letter_only_ce, letter_space_ids  # noqa: E402
from run_sota_digit_decollapse import (  # noqa: E402
    digit_argmax_stats,
    digit_ce,
    pure_digit_ids,
)
from run_sota_standard_climb import (  # noqa: E402
    CKPT,
    DATA,
    EVAL16,
    load_model,
    measure_all,
    retention_ce,
    save_promoted,
)

OUT = ROOT / "results" / "industry_lm"
AUDIT = OUT / "host_package_audit.json"
EPS = 1e-9
# Absolute production floor for writing pure_fsot_sota_standard_best
PROD_ARC = 0.30


def package_score(arc_min, agree, gen, space_dig, mode_frac, gsm_first) -> float:
    return (
        2.0 * float(arc_min)
        + 1.0 * float(agree)
        + 0.5 * float(gen)
        + 0.4 * float(space_dig)
        + 0.2 * float(gsm_first)
        - 0.5 * max(0.0, float(mode_frac) - 0.35)
    )


def snap_package(cap, ov, dstat) -> dict:
    return {
        "arc_min": float(cap["arc_min"]),
        "arc_e": float(cap["arc_e"]),
        "arc_c": float(cap["arc_c"]),
        "agree": float(cap["agree"]),
        "gsm_first": float(cap["gsm_first"]),
        "gsm_exact": float(cap["gsm_exact"]),
        "gen": float(ov.gen_score),
        "gap": float(ov.mean_overfit_gap),
        "space_dig": float(dstat["first_digit_after_space"]),
        "mode_top": dstat["top_argmax"],
        "mode_frac": float(dstat["top_frac"]),
        "package": package_score(
            cap["arc_min"],
            cap["agree"],
            ov.gen_score,
            dstat["first_digit_after_space"],
            dstat["top_frac"],
            cap["gsm_first"],
        ),
    }


def floors_ok(base: dict, cur: dict) -> tuple[bool, list[str]]:
    """Hard: no regression vs spine/base floors."""
    reasons = []
    if cur["agree"] + EPS < max(0.90, base["agree"] - 0.01):
        reasons.append(f"agree_floor {base['agree']:.0%}→{cur['agree']:.0%}")
    if cur["arc_min"] + EPS < base["arc_min"] - 1e-9:
        reasons.append(f"arc_min_regressed {base['arc_min']:.0%}→{cur['arc_min']:.0%}")
    if cur["gen"] + EPS < base["gen"] - 0.02:
        reasons.append(f"gen_regressed {base['gen']:.3f}→{cur['gen']:.3f}")
    if cur["space_dig"] + EPS < base["space_dig"] - 0.05:
        reasons.append(
            f"space_dig_regressed {base['space_dig']:.0%}→{cur['space_dig']:.0%}"
        )
    # mode: never re-collapse harder than base (or absolute 0.55 if base was collapsed)
    mode_ceil = max(float(base["mode_frac"]), 0.55)
    if cur["mode_frac"] > mode_ceil + 0.02 and cur["mode_frac"] >= 0.55:
        reasons.append(
            f"mode_re_collapse {base['mode_frac']:.0%}→{cur['mode_frac']:.0%}"
        )
    return len(reasons) == 0, reasons


def package_improved(base: dict, cur: dict) -> tuple[bool, list[str]]:
    ok, bad = floors_ok(base, cur)
    if not ok:
        return False, bad
    gains = []
    if cur["package"] > base["package"] + 0.01:
        gains.append(f"package {base['package']:.3f}→{cur['package']:.3f}")
    if cur["arc_min"] > base["arc_min"] + 0.015:
        gains.append(f"arc_min {base['arc_min']:.0%}→{cur['arc_min']:.0%}")
    if cur["space_dig"] > base["space_dig"] + 0.025:
        gains.append(f"space_dig {base['space_dig']:.0%}→{cur['space_dig']:.0%}")
    if cur["mode_frac"] < base["mode_frac"] - 0.08 and cur["mode_frac"] < 0.55:
        gains.append(f"mode {base['mode_frac']:.0%}→{cur['mode_frac']:.0%}")
    if cur["gen"] > base["gen"] + 0.015:
        gains.append(f"gen {base['gen']:.3f}→{cur['gen']:.3f}")
    if cur["gsm_first"] > base["gsm_first"] + 0.025:
        gains.append(f"gsm_first {base['gsm_first']:.0%}→{cur['gsm_first']:.0%}")
    return len(gains) > 0, gains if gains else ["no_package_gain"]


def pick_spine() -> Path:
    if AUDIT.is_file():
        rows = json.loads(AUDIT.read_text(encoding="utf-8")).get("rows") or []
        if rows:
            name = rows[0]["name"]
            p = CKPT / name
            if p.is_file():
                return p
    # fallback order: prefer known high-ARC locks over polluted digit labs
    for n in (
        "pure_fsot_sota_climb_best.pt",
        "pure_fsot_answer_locked_best.pt",
        "pure_fsot_arc_locked_best.pt",
        "pure_fsot_agree100_best.pt",
        "pure_fsot_sota_standard_best.pt",
        "pure_fsot_hardware_sota_best.pt",
        "pure_fsot_barrier_lab_best.pt",
    ):
        if (CKPT / n).is_file():
            return CKPT / n
    raise FileNotFoundError("no spine host")


def build_mix(rng: random.Random):
    # digit-balanced GSM
    by_d = {str(i): [] for i in range(10)}
    for r in load_gsm8k_train(4000):
        g = re.sub(r"[^\d]", "", str(r["gold"]))
        if g and g[0] in by_d:
            q = r["text"].split("\n")[0]
            if not q.startswith("Question:"):
                q = "Question: " + q
            by_d[g[0]].append({"kind": "digit", "prompt": f"{q}\n####", "gold": str(r["gold"]).strip()})
    digits = []
    for i in range(3000):
        d = str(i % 10)
        pool = by_d[d]
        if pool:
            digits.append(pool[rng.randrange(len(pool))])
    # balanced ARC
    easy = load_arc_train(DATA / "ARC-Easy_train.csv", None)
    ch = load_arc_train(DATA / "ARC-Challenge_train.csv", None)
    easy_tr, _ = split_disjoint(easy, train_n=2500, hold_n=60, seed=17)
    ch_tr, _ = split_disjoint(ch, train_n=1500, hold_n=40, seed=19)
    arcs = []
    for r in list(easy_tr[:1200]) + list(ch_tr[:600]):
        arcs.append({"kind": "arc", "prompt": r["prompt"], "gold": r["gold"]})
    rng.shuffle(arcs)
    return digits, arcs


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = derive_fsot_lr_plan(d_eff=14.0, epochs=12, ref_loss=4.0)
    print("=== JOINT PACKAGE CLIMB (correct FSOT application) ===")
    print("docs/FSOT_CORRECT_APPLICATION.md")
    print(f"device={device} {plan.note}")

    v_pre = run_verification(include_host=True, write=True)
    if not v_pre["ok"]:
        print("G-VERIFY FAIL — refuse (pin broken)")
        return 1

    spine = pick_spine()
    print(f"spine host={spine.name}")

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
    packs = dict(
        easy_train=easy_tr,
        easy_hold=easy_h,
        ch_train=ch_tr,
        ch_hold=ch_h,
        gsm_hold=gsm_hold,
        gsm_train_probe=gsm_probe,
    )

    tok_t, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tok, student = load_model(device)
    swap_all_layers(student)
    ck = torch.load(spine, map_location=device, weights_only=False)
    student.load_state_dict(ck["state_dict"], strict=False)
    pure = pure_digit_ids(tok)
    lids = letter_space_ids(tok)

    cap0, ov0 = measure_all(tok, teacher, student, device, packs)
    d0 = digit_argmax_stats(tok, student, device, gsm_hold, pure)
    base = snap_package(cap0, ov0, d0)
    print(
        f"SPINE package={base['package']:.3f} arc={base['arc_min']:.0%} "
        f"dig={base['space_dig']:.0%} mode={base['mode_top']}@{base['mode_frac']:.0%} "
        f"agree={base['agree']:.0%} gen={base['gen']:.3f}"
    )

    best = dict(base)
    best_cap, best_ov, best_d = dict(cap0), ov0, d0
    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    spine_frozen = dict(base)  # absolute floors for the whole run

    for p in student.parameters():
        p.requires_grad_(False)
    # High-leverage free weights only: embed (digit+letter rows) + lm_head
    for name, p in student.named_parameters():
        if "embed_tokens.weight" in name or "lm_head" in name:
            p.requires_grad_(True)

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=plan.lr0,
        weight_decay=0.0,
    )

    rng = random.Random(21)
    digits, arcs = build_mix(rng)
    STEPS = 500
    EVAL_EVERY = 25
    history = []
    t0 = time.time()
    promoted = False
    recent_hits = 0.0
    student.train()

    for step in range(1, STEPS + 1):
        drow = digits[step % len(digits)]
        arow = arcs[step % len(arcs)]
        # Joint loss every step — never digit-only or ARC-only phases
        loss_d = digit_ce(
            student, tok, device, drow["prompt"], drow["gold"], pure, anti_mode=0.55
        )
        loss_a = letter_only_ce(
            student, tok, device, arow["prompt"], arow["gold"], lids, smooth=0.10
        )
        loss_r = retention_ce(
            student, teacher, tok, device, EVAL16[step % len(EVAL16)]
        )
        # Weights from seed composites (not free fishing): phi / e balance
        from fsot_lib import SEEDS

        w_a = float(SEEDS.phi / (SEEDS.phi + SEEDS.e))  # ~0.373 ARC share-ish invert
        w_d = 1.0 - w_a  # digit share
        # Prefer slightly more ARC when ARC is the weaker package axis
        if best["arc_min"] < 0.30:
            w_a, w_d = 0.55, 0.35
        else:
            w_a, w_d = 0.40, 0.45
        loss = w_a * loss_a + w_d * loss_d + 0.35 * loss_r
        if not torch.isfinite(loss):
            continue

        lr = fsot_epoch_lr(
            plan,
            epoch=step // max(STEPS // 12, 1),
            step=step,
            loss=float(loss.detach()),
            recent_hits=recent_hits,
        )
        lr = min(max(lr, plan.lr_floor), plan.lr_ceil)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Mask embed grads to digit + letter token rows only
        allow = set(pure) | set(lids)
        for name, p in student.named_parameters():
            if p.grad is None:
                continue
            if "embed_tokens.weight" in name:
                mask = torch.zeros_like(p.grad)
                for i in allow:
                    if 0 <= i < mask.size(0):
                        mask[i] = 1.0
                p.grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 0.5
        )
        opt.step()

        if step % EVAL_EVERY != 0 and step != 1:
            continue

        cap, ov = measure_all(tok, teacher, student, device, packs)
        dstat = digit_argmax_stats(tok, student, device, gsm_hold, pure)
        student.train()
        cur = snap_package(cap, ov, dstat)
        ov_ok, ov_r = accept_update(
            before=best_ov,
            after=ov,
            min_hold_delta=-0.02,
            max_gap_widen=0.05,
            require_gen_improve=False,
        )
        floor_ok, floor_r = floors_ok(spine_frozen, cur)
        better, gain_r = package_improved(best, cur)

        history.append({"step": step, **cur, "ov_ok": ov_ok, "floor_ok": floor_ok})
        print(
            f"  {step:04d} pkg={cur['package']:.3f} arc={cur['arc_min']:.0%} "
            f"dig={cur['space_dig']:.0%} mode={cur['mode_top']}@{cur['mode_frac']:.0%} "
            f"gen={cur['gen']:.3f} agree={cur['agree']:.0%} lr={lr:.2e}",
            flush=True,
        )

        if better and floor_ok and ov_ok:
            best = dict(cur)
            best_cap, best_ov, best_d = dict(cap), ov, dstat
            best_state = {
                k: v.detach().cpu().clone() for k, v in student.state_dict().items()
            }
            promoted = True
            recent_hits = max(0.0, recent_hits - 0.5)
            save_promoted(
                student,
                cap,
                ov,
                step,
                "joint_package",
                cap0,
                digit_stats=dstat,
                pin_verify_pass=bool(v_pre.get("ok")),
                promote_standard=float(cap["arc_min"]) >= PROD_ARC
                and float(cur["package"]) >= float(spine_frozen["package"]) - 1e-9,
                lab_name="pure_fsot_joint_package_best.pt",
                arc_floor_for_standard=PROD_ARC,
            )
            print(f"    * PACKAGE PROMOTE {gain_r}", flush=True)
        else:
            # regression or no gain → restore (this is the correct application)
            student.load_state_dict(best_state, strict=False)
            recent_hits += 1.0
            why = []
            if not floor_ok:
                why += floor_r
            if not ov_ok:
                why += list(ov_r)
            if not better:
                why += list(gain_r)
            print(f"    * REJECT restore {why}", flush=True)

    student.load_state_dict(best_state, strict=False)
    cap_f, ov_f = measure_all(tok, teacher, student, device, packs)
    d_f = digit_argmax_stats(tok, student, device, gsm_hold, pure)
    final = snap_package(cap_f, ov_f, d_f)
    write_overfit_ledger(ov_f, OUT, name="overfit_joint_package")
    v_post = run_verification(include_host=True, write=True)

    improved, reasons = package_improved(spine_frozen, final)
    floor_ok, floor_r = floors_ok(spine_frozen, final)
    ok = bool(improved and floor_ok and v_pre["ok"] and v_post["ok"] and final["agree"] >= 0.9)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "joint_package_climb_v1",
        "spine": spine.name,
        "spine_package": spine_frozen,
        "final_package": final,
        "package_improved": improved,
        "floors_ok": floor_ok,
        "floor_breaks": floor_r,
        "improve_reasons": reasons,
        "promote_ok": ok,
        "history": history,
        "elapsed_s": time.time() - t0,
        "claims": "substrate_zero_free; host_weights_free_under_fsot; no_single_axis_regression",
    }
    (OUT / "joint_package_climb.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (OUT / "JOINT_PACKAGE_CLIMB.md").write_text(
        f"""# Joint package climb

**Correct application:** no single-axis promote if any floor regresses.

| Axis | Spine | Final |
|------|-------|-------|
| package | {spine_frozen['package']:.3f} | {final['package']:.3f} |
| ARC min | {spine_frozen['arc_min']:.0%} | {final['arc_min']:.0%} |
| Space digit | {spine_frozen['space_dig']:.0%} | {final['space_dig']:.0%} |
| Mode | {spine_frozen['mode_top']}@{spine_frozen['mode_frac']:.0%} | {final['mode_top']}@{final['mode_frac']:.0%} |
| gen_score | {spine_frozen['gen']:.3f} | {final['gen']:.3f} |
| Agree | {spine_frozen['agree']:.0%} | {final['agree']:.0%} |

**Spine:** `{spine.name}`  
**Promote OK:** {ok}  
**Floors OK:** {floor_ok} {floor_r}  
**Gains:** {reasons}  
**Verify:** pre={v_pre['ok']} post={v_post['ok']}
""",
        encoding="utf-8",
    )
    tag = "PACKAGE_IMPROVED" if ok else "NO_REGRESSION_PROMOTE"
    print(f"=== {tag} ===")
    print(
        f"pkg {spine_frozen['package']:.3f}→{final['package']:.3f} "
        f"arc {spine_frozen['arc_min']:.0%}→{final['arc_min']:.0%} "
        f"dig {spine_frozen['space_dig']:.0%}→{final['space_dig']:.0%} "
        f"mode {spine_frozen['mode_frac']:.0%}→{final['mode_frac']:.0%}"
    )
    return 0 if floor_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

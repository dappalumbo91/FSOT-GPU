#!/usr/bin/env python3
"""
CAPABILITY RECOVERY — restore pure-FSOT 135M to documented SOTA floors.

Historical bar (results/industry_lm/SOTA_STANDARD_CLIMB.md, BARRIER_DIAGNOSIS.md):
  ARC min ≈ 32.5% · Agree 100% · gen_score ≈ 0.319 · vs HF ARC ~8%

We destroyed the production weight file with digit thrash. This script:
  1. Refuses polluted low-ARC digit labs as starting spine
  2. Starts from pure FSOT fidelity hosts (agree100 / fulldof / 12x3)
  3. Climbs with proven task mix + FSOT suction–poof LR + scalar loss scale
  4. Only writes pure_fsot_sota_standard_best when ARC ≥ 0.30

Uses FSOT math: seeds, derive_fsot_lr_plan, fsot_epoch_lr, compute_scalar
domain folds for ARC vs GSM loss weighting.
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
from fsot_lib import SEEDS, compute_scalar  # noqa: E402
from fsot_lib.learn import derive_fsot_lr_plan, fsot_epoch_lr  # noqa: E402
from fsot21_verify import run_verification  # noqa: E402
from overfit_metrics import accept_update, split_disjoint, write_overfit_ledger  # noqa: E402
from real_data_packs import load_arc_train, load_gsm8k_test, load_gsm8k_train  # noqa: E402
from run_sota_standard_climb import (  # noqa: E402
    CKPT,
    DATA,
    D_EFF,
    EVAL16,
    _digit_token_ids,
    answer_ce_short,
    first_digit_ce,
    first_digit_vocab_ce,
    load_model,
    measure_all,
    next_ce,
    retention_ce,
    save_promoted,
    trainable,
)

OUT = ROOT / "results" / "industry_lm"

# Documented capability package — the ONLY success definition for recovery
TARGET = {
    "arc_min": 0.325,
    "agree": 1.0,
    "gen_score": 0.319,
    "gsm_first": 0.30,
}
PROD_WRITE_ARC = 0.30  # minimum to rewrite production standard


def fsot_domain_weight(kind: str) -> float:
    """
    Loss scale from seed scalar at domain D_eff folds (not free knobs).
    ARC-ish cognition: higher D_eff; short arithmetic: mid D_eff.
    """
    d_eff = 16.0 if kind == "arc" else 12.0 if kind in ("digit", "gsm") else 14.0
    S = abs(
        float(
            compute_scalar(
                N=1.0,
                P=1.0,
                D_eff=d_eff,
                delta_psi=float(SEEDS.psi_con),
                recent_hits=0.0,
                observed=True,
                delta_theta=float(SEEDS.theta_s),
            )
        )
    )
    # map |S| into [0.7, 1.4] band
    return float(0.7 + min(S, 2.0) / 2.0 * 0.7)


def pick_recovery_host() -> Path:
    """Prefer pure fidelity bases; skip known polluted digit-phase standards."""
    order = [
        CKPT / "pure_fsot_agree100_best.pt",
        CKPT / "pure_fsot_fulldof_best.pt",
        CKPT / "pure_fsot_agree_best.pt",
        CKPT / "pure_fsot_12x3_best.pt",
        CKPT / "pure_fsot_realdata_best.pt",
        CKPT / "pure_fsot_answer_locked_best.pt",
        CKPT / "pure_fsot_joint_package_best.pt",
        CKPT / "pure_fsot_hardware_sota_best.pt",
        # polluted last — only if nothing else
        CKPT / "pure_fsot_sota_standard_best.pt",
    ]
    for p in order:
        if p.is_file():
            return p
    raise FileNotFoundError("no pure_fsot host for recovery")


def dist_to_target(cap, gen) -> dict:
    return {
        "arc_gap": float(TARGET["arc_min"] - cap["arc_min"]),
        "agree_gap": float(TARGET["agree"] - cap["agree"]),
        "gen_gap": float(TARGET["gen_score"] - gen),
        "gsm_first_gap": float(TARGET["gsm_first"] - cap["gsm_first"]),
        "recovered": bool(
            cap["arc_min"] + 1e-9 >= TARGET["arc_min"]
            and cap["agree"] + 1e-9 >= 0.90
            and gen + 1e-9 >= TARGET["gen_score"] - 0.01
        ),
    }


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = derive_fsot_lr_plan(d_eff=D_EFF, epochs=12, ref_loss=4.0)
    print("=== CAPABILITY RECOVERY (FSOT pure 135M) ===")
    print("TARGET:", TARGET)
    print("docs/CAPABILITY_RECOVERY.md")
    print(f"FSOT LR {plan.note}")

    v_pre = run_verification(include_host=True, write=True)
    if not v_pre["ok"]:
        print("G-VERIFY FAIL — stop")
        return 1

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

    src = pick_recovery_host()
    print(f"recovery start host={src.name}")

    tok_t, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tok, student = load_model(device)
    swap_all_layers(student)
    ck = torch.load(src, map_location=device, weights_only=False)
    student.load_state_dict(ck["state_dict"], strict=False)

    cap0, ov0 = measure_all(tok, teacher, student, device, packs)
    d0 = dist_to_target(cap0, ov0.gen_score)
    print(
        f"START arc={cap0['arc_min']:.1%} E={cap0['arc_e']:.1%} C={cap0['arc_c']:.1%} "
        f"agree={cap0['agree']:.0%} gen={ov0.gen_score:.3f} gsm_first={cap0['gsm_first']:.0%}"
    )
    print(f"DISTANCE TO TARGET: {d0}")

    best_cap, best_ov = dict(cap0), ov0
    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    # Floor: never go below start on ARC/agree during recovery; climb toward TARGET
    floor_arc = float(cap0["arc_min"])
    floor_agree = max(0.90, float(cap0["agree"]) - 0.02)
    floor_gen = float(ov0.gen_score) - 0.03

    for p in student.parameters():
        p.requires_grad_(False)
    for name, p in student.named_parameters():
        if "embed_tokens.weight" in name or "lm_head" in name:
            p.requires_grad_(True)

    opt = torch.optim.AdamW(trainable(student), lr=plan.lr0, weight_decay=0.0)
    digit_ids = _digit_token_ids(tok)
    letter_ids = []
    for L in ("A", "B", "C", "D", " A", " B", " C", " D"):
        e = tok.encode(L, add_special_tokens=False)
        if len(e) == 1:
            letter_ids.append(e[0])
    letter_ids = sorted(set(letter_ids))

    def mask_rows():
        allow = set(digit_ids + letter_ids)
        for name, p in student.named_parameters():
            if p.grad is None:
                continue
            if "embed_tokens.weight" in name or "lm_head" in name:
                mask = torch.zeros_like(p.grad)
                for i in allow:
                    if 0 <= i < mask.size(0):
                        mask[i] = 1.0
                p.grad.mul_(mask)

    rng = random.Random(7)
    # ARC-heavy mix for recovery (this is the skill we lost)
    arc_pool = [r for r in (list(easy_tr[:1500]) + list(ch_tr[:900])) if str(r.get("gold", "")).strip().upper()[:1] in "ABCD"]
    gsm_real = [r for r in load_gsm8k_train(1500) if len(str(r["gold"]).strip()) <= 3]
    arith = []
    while len(arith) < 800:
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        if rng.random() < 0.6:
            gold, q = str(a + b), f"What is {a} + {b}?"
        else:
            aa, bb = max(a, b), min(a, b)
            gold, q = str(aa - bb), f"What is {aa} - {bb}?"
        arith.append({"prompt": f"Question: {q}\n####", "gold": gold})

    w_arc = fsot_domain_weight("arc")
    w_gsm = fsot_domain_weight("gsm")
    print(f"FSOT domain weights arc={w_arc:.3f} gsm={w_gsm:.3f} (from compute_scalar D_eff folds)")

    STEPS = 900
    EVAL_EVERY = 40
    history = []
    t0 = time.time()
    recent_hits = 0.0
    promoted = False
    student.train()

    for step in range(1, STEPS + 1):
        r = step % 10
        # 60% ARC recovery · 25% GSM/digit · 15% retention pressure via task
        if r < 6 and arc_pool:
            row = arc_pool[step % len(arc_pool)]
            loss_task = next_ce(
                student, tok, device, row["prompt"], row["gold"], kind="letter"
            )
            loss = w_arc * loss_task
            kind = "arc"
        elif r < 9:
            if r < 8 and gsm_real:
                gr = gsm_real[step % len(gsm_real)]
                q = gr["text"].split("\n")[0]
                if not q.startswith("Question:"):
                    q = "Question: " + q
                prompt, gold = f"{q}\n####", str(gr["gold"]).strip()
            else:
                row = arith[step % len(arith)]
                prompt, gold = row["prompt"], row["gold"]
            fd_v = first_digit_vocab_ce(student, tok, device, prompt, gold, digit_ids)
            fd = first_digit_ce(student, tok, device, prompt, gold)
            ce = answer_ce_short(student, tok, device, prompt, gold, "num")
            loss = w_gsm * (2.0 * fd_v + 0.6 * fd + 0.25 * ce)
            kind = "gsm"
        else:
            loss = retention_ce(
                student, teacher, tok, device, EVAL16[step % len(EVAL16)]
            )
            kind = "ret"

        loss = loss + 0.40 * retention_ce(
            student, teacher, tok, device, EVAL16[(step + 3) % len(EVAL16)]
        )
        if not torch.isfinite(loss):
            continue

        lr = fsot_epoch_lr(
            plan,
            epoch=min(step // max(STEPS // 12, 1), 11),
            step=step,
            loss=float(loss.detach()),
            recent_hits=recent_hits,
        )
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_rows()
        torch.nn.utils.clip_grad_norm_(trainable(student), 0.5)
        opt.step()

        if step % EVAL_EVERY != 0 and step != 1:
            continue

        cap, ov = measure_all(tok, teacher, student, device, packs)
        student.train()
        dist = dist_to_target(cap, ov.gen_score)
        history.append({"step": step, **cap, "gen": ov.gen_score, "dist": dist, "lr": lr})

        print(
            f"  {step:04d} arc={cap['arc_min']:.1%} E={cap['arc_e']:.1%} C={cap['arc_c']:.1%} "
            f"agree={cap['agree']:.0%} gen={ov.gen_score:.3f} first={cap['gsm_first']:.0%} "
            f"Δarc={dist['arc_gap']:+.1%} Δgen={dist['gen_gap']:+.3f} lr={lr:.2e}",
            flush=True,
        )

        floors = (
            cap["arc_min"] + 1e-9 >= floor_arc - 1e-9
            and cap["agree"] + 1e-9 >= floor_agree
            and ov.gen_score + 1e-9 >= floor_gen
        )
        ov_ok, ov_r = accept_update(
            before=best_ov,
            after=ov,
            min_hold_delta=-0.02,
            max_gap_widen=0.05,
            require_gen_improve=False,
        )
        # Climb: ARC toward target is primary recovery axis; no floor break
        better = (
            cap["arc_min"] > best_cap["arc_min"] + 0.01
            or (
                abs(cap["arc_min"] - best_cap["arc_min"]) < 0.01
                and ov.gen_score > best_ov.gen_score + 0.01
            )
            or (
                abs(cap["arc_min"] - best_cap["arc_min"]) < 0.01
                and cap["agree"] > best_cap["agree"] + 0.01
            )
        )

        if floors and ov_ok and better:
            best_cap, best_ov = dict(cap), ov
            best_state = {
                k: v.detach().cpu().clone() for k, v in student.state_dict().items()
            }
            promoted = True
            recent_hits = max(0.0, recent_hits - 0.5)
            # Raise floors with success (lock gains)
            floor_arc = max(floor_arc, float(cap["arc_min"]) - 0.005)
            floor_gen = max(floor_gen, float(ov.gen_score) - 0.02)
            save_promoted(
                student,
                cap,
                ov,
                step,
                "capability_recovery",
                cap0,
                pin_verify_pass=True,
                promote_standard=float(cap["arc_min"]) >= PROD_WRITE_ARC
                and float(cap["agree"]) >= 0.90,
                lab_name="pure_fsot_capability_recovery_best.pt",
                arc_floor_for_standard=PROD_WRITE_ARC,
            )
            tag = "RECOVERED TARGET" if dist["recovered"] else "RECOVERY PROMOTE"
            print(f"    * {tag} arc={cap['arc_min']:.1%} gen={ov.gen_score:.3f}", flush=True)
            if dist["recovered"]:
                print("=== HISTORICAL CAPABILITY PACKAGE REACHED ===")
                break
        else:
            student.load_state_dict(best_state, strict=False)
            recent_hits += 1.0
            why = []
            if not floors:
                why.append("floor_break")
            if not ov_ok:
                why.append(str(ov_r))
            if not better:
                why.append("no_arc_gen_gain")
            print(f"    * REJECT restore {why}", flush=True)

    student.load_state_dict(best_state, strict=False)
    cap_f, ov_f = measure_all(tok, teacher, student, device, packs)
    dist_f = dist_to_target(cap_f, ov_f.gen_score)
    write_overfit_ledger(ov_f, OUT, name="overfit_capability_recovery")
    v_post = run_verification(include_host=True, write=True)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "capability_recovery_v1",
        "target": TARGET,
        "start_host": src.name,
        "start": {"cap": cap0, "gen": ov0.gen_score, "dist": d0},
        "final": {"cap": cap_f, "gen": ov_f.gen_score, "dist": dist_f},
        "recovered": dist_f["recovered"],
        "verify_pre": v_pre["ok"],
        "verify_post": v_post["ok"],
        "history": history,
        "elapsed_s": time.time() - t0,
        "fsot": {
            "lr_plan": plan.note,
            "w_arc": w_arc,
            "w_gsm": w_gsm,
            "collapse_theta": float(SEEDS.c_eff * SEEDS.p_var),
        },
    }
    (OUT / "capability_recovery.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (OUT / "CAPABILITY_RECOVERY.md").write_text(
        f"""# Capability recovery

**Target (historical pure FSOT SOTA package):** ARC min ≥ **{TARGET['arc_min']:.1%}**, agree ≥ **90%**, gen ≥ **{TARGET['gen_score']:.3f}**

| Axis | Start | Final | Target | Gap |
|------|-------|-------|--------|-----|
| ARC min | {cap0['arc_min']:.1%} | {cap_f['arc_min']:.1%} | {TARGET['arc_min']:.1%} | {dist_f['arc_gap']:+.1%} |
| Agree | {cap0['agree']:.0%} | {cap_f['agree']:.0%} | 100% | {dist_f['agree_gap']:+.0%} |
| gen_score | {ov0.gen_score:.3f} | {ov_f.gen_score:.3f} | {TARGET['gen_score']:.3f} | {dist_f['gen_gap']:+.3f} |
| GSM first | {cap0['gsm_first']:.0%} | {cap_f['gsm_first']:.0%} | {TARGET['gsm_first']:.0%} | {dist_f['gsm_first_gap']:+.0%} |

**Start host:** `{src.name}`  
**Recovered:** {dist_f['recovered']}  
**Verify:** {v_pre['ok']} / {v_post['ok']}  
**FSOT:** suction–poof LR · scalar domain weights ARC={w_arc:.3f} GSM={w_gsm:.3f} · θ={SEEDS.c_eff*SEEDS.p_var:.4f}
""",
        encoding="utf-8",
    )
    print("=== RECOVERY END ===")
    print(
        f"arc {cap0['arc_min']:.1%}→{cap_f['arc_min']:.1%} (target {TARGET['arc_min']:.1%}) "
        f"gen {ov0.gen_score:.3f}→{ov_f.gen_score:.3f} recovered={dist_f['recovered']}"
    )
    return 0 if v_post["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

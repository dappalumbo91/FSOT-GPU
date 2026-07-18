#!/usr/bin/env python3
"""Short balanced follow-up: protect ARC-Challenge + single-digit arith."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from fsot_layer_swap import swap_all_layers  # noqa: E402
from fsot_lib.learn import derive_fsot_lr_plan, fsot_epoch_lr  # noqa: E402
from granular_metrics import agree_n, eval_arc_granular, eval_gsm_granular  # noqa: E402
from real_data_packs import load_gsm8k_test  # noqa: E402
from run_granular_push import (  # noqa: E402
    CKPT,
    DATA,
    EVAL16,
    answer_ce,
    digit_after_space_ce,
    load_model,
    next_ce,
    retention_ce,
    split_arc,
)

STEPS = 500


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = derive_fsot_lr_plan(d_eff=14.0, epochs=12, ref_loss=4.0)
    tok, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tok, student = load_model(device)
    swap_all_layers(student)
    src = CKPT / "pure_fsot_granular_best.pt"
    ck = torch.load(src, map_location=device, weights_only=False)
    student.load_state_dict(ck["state_dict"], strict=False)
    for p in student.parameters():
        p.requires_grad_(True)
    print("loaded", src.name, "step", ck.get("step"))

    rng = random.Random(5)
    arith = []
    for _ in range(800):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        if rng.random() < 0.5:
            gold = str(a + b)
            q = f"What is {a} + {b}?"
        else:
            aa, bb = max(a, b), min(a, b)
            gold = str(aa - bb)
            q = f"What is {aa} - {bb}?"
        arith.append({"prompt": f"Question: {q}\n####", "gold": gold})

    arc_e, hold_e = split_arc(DATA / "ARC-Easy_train.csv", 2500, 60, 17)
    arc_c, hold_c = split_arc(DATA / "ARC-Challenge_train.csv", 1500, 40, 19)
    # overweight challenge in train mix
    arc_train = arc_e + arc_c + arc_c
    random.Random(7).shuffle(arc_train)

    gsm_eval = load_gsm8k_test(40)
    for r in gsm_eval:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"

    opt = torch.optim.AdamW(list(student.parameters()), lr=plan.lr0 * 0.35, weight_decay=0.01)
    best_score = -1.0
    best_arc = None
    student.train()

    for step in range(1, STEPS + 1):
        if step % 5 == 0:
            row = arith[step % len(arith)]
            ce = answer_ce(student, tok, device, row["prompt"], row["gold"], "num")
            ce1 = next_ce(student, tok, device, row["prompt"], row["gold"], "num")
            ce_d = digit_after_space_ce(student, tok, device, row["prompt"], row["gold"])
            loss_task = ce + ce1 + ce_d
        else:
            row = arc_train[step % len(arc_train)]
            gold = row["gold"].strip().upper()[:1]
            if gold not in "ABCD":
                continue
            ce = answer_ce(student, tok, device, row["prompt"], gold, "letter")
            ce1 = next_ce(student, tok, device, row["prompt"], gold, "letter")
            loss_task = ce + 0.8 * ce1
        ce_r = retention_ce(student, teacher, tok, device, EVAL16[step % len(EVAL16)])
        loss = loss_task + 0.4 * ce_r
        if not torch.isfinite(loss):
            continue
        lr = min(
            fsot_epoch_lr(
                plan,
                epoch=min(step // 50, 11),
                step=step,
                loss=float(loss.item()),
                recent_hits=0.0,
            )
            * 0.4,
            plan.lr0 * 0.4,
        )
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()), 0.5)
        opt.step()

        if step % 100 == 0 or step == 1:
            student.eval()
            g, _ = eval_gsm_granular(tok, student, device, gsm_eval, arm="fsot")
            ae, _ = eval_arc_granular(tok, student, device, hold_e, arm="fsot")
            ac, _ = eval_arc_granular(tok, student, device, hold_c, arm="fsot")
            ag = agree_n(tok, teacher, student, device, EVAL16)
            sc = (
                2.0 * ae["exact"]
                + 1.5 * ac["exact"]
                + 1.0 * (g["first_digit"] or 0)
                + 0.5 * (g["tf_token_acc"] or 0)
                + 0.4 * ag
            )
            student.train()
            print(
                f"{step:04d} arcE={ae['exact']:.0%} arcC={ac['exact']:.0%} "
                f"gsm_first={g['first_digit']:.0%} gsm_tf={g['tf_token_acc']:.0%} "
                f"gsm_x={g['exact']:.0%} ag={ag:.0%} sc={sc:.3f} mode={g.get('mode_pred')}"
            )
            if sc > best_score and ag >= 0.9:
                best_score = sc
                best_arc = (ae["exact"], ac["exact"])
                torch.save(
                    {
                        "state_dict": {
                            k: v.detach().cpu() for k, v in student.state_dict().items()
                        },
                        "step": step,
                        "arc_easy_hold": ae["exact"],
                        "arc_challenge_hold": ac["exact"],
                        "gsm_first": g["first_digit"],
                        "gsm_exact": g["exact"],
                        "agree16": ag,
                        "balanced_score": sc,
                        "granular_push": True,
                        "phase": "granular_balance",
                    },
                    CKPT / "pure_fsot_granular_best.pt",
                )
                print("  * BEST", best_arc, "sc", sc)

    print("done best", best_arc, best_score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

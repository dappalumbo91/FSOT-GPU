#!/usr/bin/env python3
"""
SOTA standards — ARC-only soft climb (no GSM digit CE).

Goal: lift arc_min / both holds under G-VERIFY + G-OVERFIT without
touching digit logits. Letter-row mask only on tied embed.
"""
from __future__ import annotations

import json
import random
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

from fsot_layer_swap import swap_all_layers  # noqa: E402
from fsot_lib.learn import derive_fsot_lr_plan  # noqa: E402
from fsot21_verify import run_verification  # noqa: E402
from overfit_metrics import accept_update, direction_label, write_overfit_ledger  # noqa: E402
from real_data_packs import load_arc_train  # noqa: E402
from run_sota_standard_climb import (  # noqa: E402
    CKPT,
    DATA,
    EVAL16,
    capability_improve,
    load_model,
    measure_all,
    next_ce,
    retention_ce,
    save_promoted,
)
from overfit_metrics import split_disjoint  # noqa: E402
from real_data_packs import load_gsm8k_test, load_gsm8k_train  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = derive_fsot_lr_plan(d_eff=14.0, epochs=12, ref_loss=4.0)
    print("=== SOTA ARC-ONLY CLIMB ===")

    v_pre = run_verification(include_host=True, write=True)
    print("verify_pre", v_pre["ok"])
    if not v_pre["ok"]:
        return 1

    easy_all = load_arc_train(DATA / "ARC-Easy_train.csv", None)
    ch_all = load_arc_train(DATA / "ARC-Challenge_train.csv", None)
    easy_tr, easy_h = split_disjoint(easy_all, train_n=2500, hold_n=60, seed=17)
    ch_tr, ch_h = split_disjoint(ch_all, train_n=1500, hold_n=40, seed=19)
    gsm_hold = load_gsm8k_test(40)
    for r in gsm_hold:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"
    gsm_tr_raw = load_gsm8k_train(200)
    gsm_train_probe = []
    for r in gsm_tr_raw:
        q = r["text"].split("\n")[0]
        if not q.startswith("Question:"):
            q = "Question: " + q
        gsm_train_probe.append({"prompt": f"{q}\n####", "gold": r["gold"]})
    packs = dict(
        easy_train=easy_tr,
        easy_hold=easy_h,
        ch_train=ch_tr,
        ch_hold=ch_h,
        gsm_hold=gsm_hold,
        gsm_train_probe=gsm_train_probe,
    )

    tok_t, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    src = CKPT / "pure_fsot_sota_standard_best.pt"
    tok, student = load_model(device)
    swap_all_layers(student)
    ck0 = torch.load(src, map_location=device, weights_only=False)
    student.load_state_dict(ck0["state_dict"], strict=False)
    print("host", src.name)

    cap0, ov0 = measure_all(tok, teacher, student, device, packs)
    print(
        f"START min={cap0['arc_min']:.0%} E={cap0['arc_e']:.0%} C={cap0['arc_c']:.0%} "
        f"gen={ov0.gen_score:.3f}"
    )
    best_cap, best_ov = dict(cap0), ov0
    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    promoted = False
    history = []

    for p in student.parameters():
        p.requires_grad_(False)
    for name, p in student.named_parameters():
        if "embed_tokens.weight" in name:
            p.requires_grad_(True)
    letter_ids = []
    for L in ("A", "B", "C", "D", " A", " B", " C", " D"):
        e = tok.encode(L, add_special_tokens=False)
        if len(e) == 1:
            letter_ids.append(e[0])
    letter_ids = sorted(set(letter_ids))
    print("letter ids", letter_ids)

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=plan.lr0 * 0.35,
        weight_decay=0.0,
    )
    # balanced C-heavy if C is bottleneck
    arc_mix = ch_tr[:1000] + easy_tr[:1000] + ch_tr[:500]
    random.Random(41).shuffle(arc_mix)
    reject = 0
    t0 = time.time()
    student.train()

    for step in range(1, 501):
        row = arc_mix[step % len(arc_mix)]
        gold = row["gold"].strip().upper()[:1]
        if gold not in "ABCD":
            continue
        loss = next_ce(student, tok, device, row["prompt"], gold, "letter")
        loss = loss + 0.6 * retention_ce(
            student, teacher, tok, device, EVAL16[step % len(EVAL16)]
        )
        if not torch.isfinite(loss):
            continue
        for g in opt.param_groups:
            g["lr"] = plan.lr0 * 0.3
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # letter-row mask only
        for name, p in student.named_parameters():
            if p.grad is None:
                continue
            if "embed_tokens.weight" in name:
                mask = torch.zeros_like(p.grad)
                for i in letter_ids:
                    if 0 <= i < mask.size(0):
                        mask[i] = 1.0
                p.grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 0.4
        )
        opt.step()

        if step % 50 != 0 and step != 1:
            continue
        cap, ov = measure_all(tok, teacher, student, device, packs)
        student.train()
        ov_ok, ov_r = accept_update(
            before=best_ov, after=ov, min_hold_delta=-0.005, max_gap_widen=0.03, require_gen_improve=False
        )
        if ov.gen_score > best_ov.gen_score + 1e-4:
            ov_ok = True
            ov_r = ["gen_score_up"]
        cap_ok, cap_r = capability_improve(cap, best_cap)
        dlab = direction_label(best_ov, ov)
        history.append(
            {
                "step": step,
                **cap,
                "gen_score": ov.gen_score,
                "gap": ov.mean_overfit_gap,
                "dir": dlab,
                "ov_ok": ov_ok,
                "cap_ok": cap_ok,
            }
        )
        print(
            f"  {step:04d} min={cap['arc_min']:.0%} E={cap['arc_e']:.0%} C={cap['arc_c']:.0%} "
            f"gen={ov.gen_score:.3f} gap={ov.mean_overfit_gap:+.0%} dir={dlab} "
            f"ov={ov_ok} cap={cap_ok}"
        )
        if cap_ok and ov_ok:
            best_cap, best_ov = dict(cap), ov
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            promoted = True
            reject = 0
            save_promoted(student, cap, ov, step, "sota_arc_only", cap0)
            print("    * PROMOTED", cap_r, ov_r)
        elif not ov_ok or cap["arc_min"] + 1e-9 < best_cap["arc_min"] - 0.015:
            student.load_state_dict(best_state, strict=False)
            reject += 1
            print("    * REJECT restore", ov_r if not ov_ok else "arc_min")
            if reject >= 4:
                break
        else:
            reject = 0

    student.load_state_dict(best_state, strict=False)
    cap_f, ov_f = measure_all(tok, teacher, student, device, packs)
    write_overfit_ledger(ov_f, ROOT / "results" / "industry_lm", name="overfit_sota_arc_only")
    v_post = run_verification(
        include_host=True,
        ckpt_path=CKPT / "pure_fsot_sota_standard_best.pt",
        write=True,
    )
    cap_beat, reasons = capability_improve(cap_f, cap0)
    ov_beat, ov_r = accept_update(
        before=ov0, after=ov_f, min_hold_delta=-0.005, max_gap_widen=0.03, require_gen_improve=False
    )
    if ov_f.gen_score > ov0.gen_score + 1e-4:
        ov_beat = True
    promote = bool(promoted and cap_beat and ov_beat and v_pre["ok"] and v_post["ok"])
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "sota_arc_only_letter_row",
        "start": cap0,
        "final": cap_f,
        "start_gen": ov0.gen_score,
        "final_gen": ov_f.gen_score,
        "promote_to_github": promote,
        "history": history,
        "elapsed_s": time.time() - t0,
        "verify_pre": v_pre["ok"],
        "verify_post": v_post["ok"],
        "reasons": reasons if promote else [],
    }
    out = ROOT / "results" / "industry_lm" / "sota_arc_only.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    status = "IMPROVED — eligible to push" if promote else "NO_PUSH"
    print("===", status, "===")
    print(
        f"min {cap0['arc_min']:.0%}→{cap_f['arc_min']:.0%} "
        f"gen {ov0.gen_score:.3f}→{ov_f.gen_score:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

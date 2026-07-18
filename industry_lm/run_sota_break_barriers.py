#!/usr/bin/env python3
"""
Break diagnosed barriers under SOTA standards.

From barrier_diagnosis.json:
  1) GSM_FREE_GEN_COLLAPSE — mode 1200000@80%; TF first ~100% vs free first 30%
  2) ARC_LETTER_MODE_COLLAPSE — free-gen ~80% letter D
  3) EVAL_NOISE — boot halfwidth ~9% (promote only clear deltas)

Levers:
  ARC: letter-only softmax CE + label smoothing + class-balanced sampling
  GSM: digit-only softmax CE after '#### ' (matches TF path that already works)
  Never multi-task destroy: alternate steps; letter-row / digit-row grad masks
  Promote: G-VERIFY + overfit accept + capability improve (prefer first-digit or arc_min)
"""
from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from fsot_layer_swap import swap_all_layers  # noqa: E402
from fsot_lib.learn import derive_fsot_lr_plan  # noqa: E402
from fsot21_verify import run_verification  # noqa: E402
from granular_metrics import constrained_digit_gen, digit_token_ids  # noqa: E402
from overfit_metrics import accept_update, direction_label, write_overfit_ledger  # noqa: E402
from overfit_metrics import split_disjoint  # noqa: E402
from real_data_packs import load_arc_train, load_gsm8k_test, load_gsm8k_train  # noqa: E402
from run_sota_standard_climb import (  # noqa: E402
    CKPT,
    DATA,
    EVAL16,
    capability_improve,
    load_model,
    measure_all,
    retention_ce,
    save_promoted,
)

OUT = ROOT / "results" / "industry_lm"


def letter_space_ids(tok):
    """Prefer single-token ' A'..' D' forms used after Answer:"""
    ids = []
    for L in (" A", " B", " C", " D"):
        e = tok.encode(L, add_special_tokens=False)
        if len(e) == 1:
            ids.append(e[0])
    return ids  # order A,B,C,D


def digit_ids(tok):
    return digit_token_ids(tok)[:10]  # 0-9 only, drop space/nl/eos if present


def letter_only_ce(student, tok, device, prompt, gold, letter_ids, smooth=0.15):
    pe = tok(prompt, return_tensors="pt", truncation=True, max_length=400).to(device)
    logits = student(**pe).logits[0, -1].float()
    sub = torch.stack([logits[i] for i in letter_ids], dim=0).unsqueeze(0)
    # gold A=0..D=3
    gi = "ABCD".index(gold.upper()[:1])
    return F.cross_entropy(sub, torch.tensor([gi], device=device), label_smoothing=smooth)


def digit_only_ce(student, tok, device, prompt, gold, dig_ids):
    g = str(gold).strip().replace(",", "")
    m = re.search(r"\d", g)
    if not m:
        return torch.tensor(0.0, device=device)
    d = int(m.group(0))
    # force space then digit-only ranking (TF path that scores ~100% first)
    pe = tok(prompt + " ", return_tensors="pt", truncation=True, max_length=400).to(device)
    logits = student(**pe).logits[0, -1].float()
    # dig_ids may include space — use pure 0-9
    pure = []
    for ch in "0123456789":
        e = tok.encode(ch, add_special_tokens=False)
        if len(e) == 1:
            pure.append(e[0])
    sub = torch.stack([logits[i] for i in pure], dim=0).unsqueeze(0)
    return F.cross_entropy(sub, torch.tensor([d], device=device))


def mask_rows(student, allow_ids):
    for name, p in student.named_parameters():
        if p.grad is None:
            continue
        if "embed_tokens.weight" not in name:
            continue
        mask = torch.zeros_like(p.grad)
        for i in allow_ids:
            if 0 <= i < mask.size(0):
                mask[i] = 1.0
        p.grad.mul_(mask)


def balanced_arc(rows):
    by = defaultdict(list)
    for r in rows:
        g = r["gold"].strip().upper()[:1]
        if g in "ABCD":
            by[g].append(r)
    # round-robin
    out = []
    keys = list("ABCD")
    idx = {k: 0 for k in keys}
    for _ in range(2000):
        k = keys[_ % 4]
        if not by[k]:
            continue
        out.append(by[k][idx[k] % len(by[k])])
        idx[k] += 1
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = derive_fsot_lr_plan(d_eff=14.0, epochs=12, ref_loss=4.0)
    print("=== SOTA BREAK BARRIERS ===")
    print("targets: ARC letter collapse + GSM free-gen collapse")

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
    gsm_tr_raw = load_gsm8k_train(500)
    gsm_probe = []
    for r in gsm_tr_raw:
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

    src = CKPT / "pure_fsot_sota_standard_best.pt"
    tok, student = load_model(device)
    swap_all_layers(student)
    ck = torch.load(src, map_location=device, weights_only=False)
    student.load_state_dict(ck["state_dict"], strict=False)
    print("host", src.name)

    lid = letter_space_ids(tok)
    print("letter ids A-D", lid)
    assert len(lid) == 4

    cap0, ov0 = measure_all(tok, teacher, student, device, packs)
    # constrained GSM first-digit (diagnostic at start)
    pure_dig = [tok.encode(d, add_special_tokens=False)[0] for d in "0123456789"]
    const_hits = 0
    for r in gsm_hold:
        th = constrained_digit_gen(
            tok, student, device, r["prompt"], max_new=6, allow_ids=pure_dig + [tok.encode(" ", add_special_tokens=False)[0]]
        )
        nums = re.findall(r"-?\d+", th.replace(",", ""))
        pred = nums[0] if nums else None
        gold = re.findall(r"-?\d+", str(r["gold"]).replace(",", ""))
        gold = gold[-1] if gold else str(r["gold"]).strip()
        if pred and gold and pred[0] == gold[0]:
            const_hits += 1
    print(
        f"START min={cap0['arc_min']:.0%} first={cap0['gsm_first']:.0%} "
        f"tf={cap0['gsm_tf']:.0%} gen={ov0.gen_score:.3f} "
        f"constrained_first≈{const_hits/len(gsm_hold):.0%}"
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

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=plan.lr0 * 0.45,
        weight_decay=0.0,
    )

    arc_bal = balanced_arc(easy_tr + ch_tr)
    # short arith for digits
    rng = random.Random(7)
    arith = []
    while len(arith) < 1500:
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        if rng.random() < 0.55:
            gold, q = str(a + b), f"What is {a} + {b}?"
        else:
            aa, bb = max(a, b), min(a, b)
            gold, q = str(aa - bb), f"What is {aa} - {bb}?"
        if len(gold) <= 2:
            arith.append({"prompt": f"Question: {q}\n####", "gold": gold})
    gsm_real = [r for r in load_gsm8k_train(2000) if len(str(r["gold"]).strip()) <= 3]
    random.Random(8).shuffle(gsm_real)

    reject = 0
    t0 = time.time()
    student.train()
    STEPS = 600

    for step in range(1, STEPS + 1):
        r = step % 10
        # 50% ARC letter-only, 50% GSM digit-only (separate masks)
        if r < 5:
            row = arc_bal[step % len(arc_bal)]
            gold = row["gold"].strip().upper()[:1]
            loss_task = letter_only_ce(
                student, tok, device, row["prompt"], gold, lid, smooth=0.2
            )
            allow = lid
            task = "arc_letter"
        else:
            if r % 2 == 0:
                row = arith[step % len(arith)]
                prompt, gold = row["prompt"], row["gold"]
            else:
                row = gsm_real[step % len(gsm_real)]
                q = row["text"].split("\n")[0]
                if not q.startswith("Question:"):
                    q = "Question: " + q
                prompt, gold = f"{q}\n####", str(row["gold"]).strip()
            loss_task = digit_only_ce(student, tok, device, prompt, gold, pure_dig)
            allow = pure_dig
            task = "gsm_digit"

        ce_r = retention_ce(student, teacher, tok, device, EVAL16[step % len(EVAL16)])
        loss = loss_task + 0.5 * ce_r
        if not torch.isfinite(loss):
            continue
        for g in opt.param_groups:
            g["lr"] = plan.lr0 * 0.4
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_rows(student, allow)
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 0.4
        )
        opt.step()

        if step % 50 != 0 and step != 1:
            continue

        cap, ov = measure_all(tok, teacher, student, device, packs)
        student.train()
        # letter collapse monitor
        # light: use gen from measure_all history only via free_gen sample on 20
        from collections import Counter as _Counter

        from granular_metrics import eval_arc_granular

        student.eval()
        ae, items = eval_arc_granular(tok, student, device, easy_h, arm="mon")
        preds = [it["pred"] for it in items if it.get("pred") in list("ABCD")]
        if preds:
            top_frac = max(_Counter(preds).values()) / len(preds)
        else:
            top_frac = 0.0
        student.train()

        ov_ok, ov_r = accept_update(
            before=best_ov,
            after=ov,
            min_hold_delta=-0.01,
            max_gap_widen=0.04,
            require_gen_improve=False,
        )
        if ov.gen_score > best_ov.gen_score + 1e-4:
            ov_ok = True
            ov_r = ["gen_score_up"]
        cap_ok, cap_r = capability_improve(cap, best_cap)
        # also promote if letter collapse eases AND arc_min holds + gen not down
        collapse_ease = top_frac < 0.65 and (best_cap.get("_top_d", 0.8) >= 0.65)
        dlab = direction_label(best_ov, ov)
        history.append(
            {
                "step": step,
                "task": task,
                **cap,
                "gen_score": ov.gen_score,
                "gap": ov.mean_overfit_gap,
                "dir": dlab,
                "easy_top_pred_frac": top_frac,
                "ov_ok": ov_ok,
                "cap_ok": cap_ok,
            }
        )
        print(
            f"  {step:04d} {task} min={cap['arc_min']:.0%} E={cap['arc_e']:.0%} "
            f"C={cap['arc_c']:.0%} first={cap['gsm_first']:.0%} tf={cap['gsm_tf']:.0%} "
            f"gen={ov.gen_score:.3f} Dfrac={top_frac:.0%} dir={dlab} ov={ov_ok} cap={cap_ok}"
        )

        if cap_ok and ov_ok:
            best_cap = dict(cap)
            best_cap["_top_d"] = top_frac
            best_ov = ov
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            promoted = True
            reject = 0
            save_promoted(student, cap, ov, step, "sota_break_barriers", cap0)
            print("    * PROMOTED", cap_r, ov_r, f"Dfrac={top_frac:.0%}")
        elif not ov_ok or dlab in ("OVERFIT_STEP", "MEMORIZE_COLLAPSE"):
            student.load_state_dict(best_state, strict=False)
            reject += 1
            print("    * REJECT", ov_r)
            if reject >= 5:
                break
        elif cap["arc_min"] + 1e-9 < best_cap["arc_min"] - 0.02:
            student.load_state_dict(best_state, strict=False)
            reject += 1
            print("    * REJECT arc_min")
            if reject >= 5:
                break
        else:
            reject = 0

    student.load_state_dict(best_state, strict=False)
    cap_f, ov_f = measure_all(tok, teacher, student, device, packs)
    write_overfit_ledger(ov_f, OUT, name="overfit_break_barriers")
    # letter dist final
    student.eval()
    _, items = eval_arc_granular(tok, student, device, easy_h, arm="final")
    preds = [it["pred"] for it in items if it.get("pred") in list("ABCD")]
    from collections import Counter

    pdist = dict(Counter(preds))
    top_f = max(pdist.values()) / max(len(preds), 1) if preds else 0

    v_post = run_verification(
        include_host=True,
        ckpt_path=CKPT / "pure_fsot_sota_standard_best.pt",
        write=True,
    )
    cap_beat, reasons = capability_improve(cap_f, cap0)
    ov_beat, ov_r = accept_update(
        before=ov0, after=ov_f, min_hold_delta=-0.01, max_gap_widen=0.04, require_gen_improve=False
    )
    if ov_f.gen_score > ov0.gen_score + 1e-4:
        ov_beat = True
    # secondary win: collapse reduced with arc_min not down and gen not down much
    collapse_win = (
        top_f < 0.65
        and cap_f["arc_min"] + 1e-9 >= cap0["arc_min"] - 0.01
        and ov_f.gen_score + 1e-9 >= ov0.gen_score - 0.02
        and cap_f["agree"] >= 0.9
    )
    promote = bool(
        promoted and (cap_beat or collapse_win) and ov_beat and v_pre["ok"] and v_post["ok"]
    )
    # stricter: only github if cap_beat (not only collapse) unless gen also up
    if collapse_win and not cap_beat and ov_f.gen_score <= ov0.gen_score:
        promote = False

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "break_barriers: letter-only+smoothing + digit-only CE row masks",
        "barriers_targeted": ["GSM_FREE_GEN_COLLAPSE", "ARC_LETTER_MODE_COLLAPSE"],
        "start": cap0,
        "final": cap_f,
        "start_gen": ov0.gen_score,
        "final_gen": ov_f.gen_score,
        "easy_pred_dist_final": pdist,
        "easy_top_pred_frac_final": top_f,
        "promote_to_github": promote,
        "capability_improved": cap_beat,
        "collapse_eased": top_f < 0.65,
        "history": history,
        "elapsed_s": time.time() - t0,
        "verify_pre": v_pre["ok"],
        "verify_post": v_post["ok"],
        "reasons": reasons if promote else [],
    }
    (OUT / "sota_break_barriers.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (OUT / "SOTA_BREAK_BARRIERS.md").write_text(
        f"""# Break barriers climb

**Status: {'IMPROVED — eligible to push' if promote else 'NO_PUSH'}**

## Diagnosed barriers (from barrier_diagnosis)
1. GSM free-gen collapse (mode 1200000)
2. ARC letter mode collapse (~80% D)

## Results

| Axis | Start | Final |
|------|-------|-------|
| ARC min | {cap0['arc_min']:.0%} | {cap_f['arc_min']:.0%} |
| GSM first | {cap0['gsm_first']:.0%} | {cap_f['gsm_first']:.0%} |
| gen_score | {ov0.gen_score:.3f} | {ov_f.gen_score:.3f} |
| Easy top-letter frac | ~80% D | {top_f:.0%} `{pdist}` |

Verify pre/post: {v_pre['ok']}/{v_post['ok']}
""",
        encoding="utf-8",
    )
    print("===", "IMPROVED" if promote else "NO_PUSH", "===")
    print(
        f"min {cap0['arc_min']:.0%}→{cap_f['arc_min']:.0%} first {cap0['gsm_first']:.0%}→{cap_f['gsm_first']:.0%} "
        f"Dfrac→{top_f:.0%} gen {ov0.gen_score:.3f}→{ov_f.gen_score:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

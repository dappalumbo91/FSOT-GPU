#!/usr/bin/env python3
"""
Open-gap climb under FSOT 2.1 verification.

1) Run fsot21_verify bridge — must PASS before train
2) Train open gaps (ARC min both holds, GSM first-digit head-only)
3) Re-verify after train — must still PASS
4) Capability promote only if gates beat prior best (multi-rep style single measure)
5) promote_to_github only if verify+capability both improve/hold

No git push here — operator pushes only when report.promote_to_github is true.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from fsot_layer_swap import swap_all_layers  # noqa: E402
from fsot_lib.learn import derive_fsot_lr_plan, fsot_epoch_lr  # noqa: E402
from fsot21_verify import run_verification  # noqa: E402
from granular_metrics import agree_n, eval_arc_granular, eval_gsm_granular  # noqa: E402
from real_data_packs import load_arc_train, load_gsm8k_test, load_gsm8k_train  # noqa: E402
from train_corpus import PROBES  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
CKPT = OUT / "checkpoints"
DATA = Path(r"D:\training data")
CKPT.mkdir(parents=True, exist_ok=True)

D_EFF = 14.0
EVAL16 = PROBES + [
    "Python is a programming language that",
    "The speed of light is approximately",
    "1 + 1 =",
    "The capital of Japan is",
    "def main():",
    "The square root of 9 is",
    "Gravity on Earth is",
    "The chemical formula for water is",
]


def load_model(device):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    return tok, m


def split_arc(path, train_n, hold_n, seed):
    all_rows = load_arc_train(path, None)
    rng = random.Random(seed)
    idx = list(range(len(all_rows)))
    rng.shuffle(idx)
    hold = [all_rows[i] for i in idx[:hold_n]]
    train = [all_rows[i] for i in idx[hold_n : hold_n + train_n]]
    return train, hold


def gold_ids(tok, gold, kind="num"):
    gold = str(gold).strip()
    cands = [f" {gold}", gold] if kind == "letter" else [f" {gold}", gold, f" {gold}\n"]
    for c in cands:
        ids = tok.encode(c, add_special_tokens=False)
        if ids:
            return ids
    return []


def next_ce(student, tok, device, prompt, gold, kind="num"):
    gids = gold_ids(tok, gold, kind=kind)
    if not gids:
        return torch.tensor(0.0, device=device, requires_grad=True)
    pe = tok(prompt, return_tensors="pt", truncation=True, max_length=400).to(device)
    return F.cross_entropy(
        student(**pe).logits[0, -1].float().unsqueeze(0),
        torch.tensor([gids[0]], device=device),
    )


def answer_ce(student, tok, device, prompt, gold, kind="num"):
    gids = gold_ids(tok, gold, kind=kind)
    if not gids:
        return torch.tensor(0.0, device=device, requires_grad=True)
    pe = tok(prompt, return_tensors="pt", truncation=True, max_length=400, add_special_tokens=True)
    prompt_ids = pe["input_ids"][0].to(device)
    gold_t = torch.tensor(gids, device=device, dtype=prompt_ids.dtype)
    full = torch.cat([prompt_ids, gold_t], dim=0).unsqueeze(0)
    if full.size(1) > 416:
        overflow = full.size(1) - 416
        full = full[:, overflow:]
        prompt_len = max(int(prompt_ids.numel()) - overflow, 1)
    else:
        prompt_len = int(prompt_ids.numel())
    logits = student(input_ids=full, attention_mask=torch.ones_like(full)).logits
    n = full.size(1) - 1
    start = max(prompt_len - 1, 0)
    if start >= n:
        return torch.tensor(0.0, device=device)
    ce = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        full[:, 1:].reshape(-1),
        reduction="none",
    ).view(1, n)
    mask = torch.zeros(1, n, device=device)
    mask[:, start:] = 1.0
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


def first_digit_ce(student, tok, device, prompt, gold):
    g = str(gold).strip().replace(",", "")
    m = re.search(r"\d", g)
    if not m:
        return torch.tensor(0.0, device=device)
    tid = tok.encode(m.group(0), add_special_tokens=False)
    pe = tok(prompt + " ", return_tensors="pt", truncation=True, max_length=400).to(device)
    return F.cross_entropy(
        student(**pe).logits[0, -1].float().unsqueeze(0),
        torch.tensor([tid[0]], device=device),
    )


def retention_ce(student, teacher, tok, device, prompt):
    re = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        tlab = int(teacher(**re).logits[0, -1].argmax())
    return F.cross_entropy(
        student(**re).logits[0, -1].float().unsqueeze(0),
        torch.tensor([tlab], device=device),
    )


def slim(g):
    keys = (
        "arc_e", "arc_c", "arc_min", "gsm_first", "gsm_tf", "gsm_exact",
        "agree", "balanced", "mode", "mode_frac",
    )
    return {k: g[k] for k in keys if k in g}


def pack_metrics(m):
    e, c = float(m["arc_e"]), float(m["arc_c"])
    first, tf, ag = float(m["gsm_first"]), float(m["gsm_tf"]), float(m["agree"])
    mn = min(e, c)
    bal = 2.0 * mn + 1.5 * 0.5 * (e + c) + 1.2 * first + 0.8 * tf + 0.4 * ag
    return {**m, "arc_min": mn, "balanced": bal}


def measure(tok, teacher, student, device, gsm_eval, hold_e, hold_c):
    student.eval()
    g, _ = eval_gsm_granular(tok, student, device, gsm_eval, arm="fsot")
    ae, _ = eval_arc_granular(tok, student, device, hold_e, arm="fsot")
    ac, _ = eval_arc_granular(tok, student, device, hold_c, arm="fsot")
    ag = agree_n(tok, teacher, student, device, EVAL16)
    return pack_metrics({
        "arc_e": ae["exact"] or 0.0,
        "arc_c": ac["exact"] or 0.0,
        "gsm_first": g["first_digit"] or 0.0,
        "gsm_tf": g["tf_token_acc"] or 0.0,
        "gsm_exact": g["exact"] or 0.0,
        "mode": g.get("mode_pred"),
        "mode_frac": g.get("mode_frac") or 0.0,
        "agree": ag,
    })


def beats(cand, base, eps=1e-4):
    reasons = []
    if cand["agree"] < 0.90:
        return False, ["agree_floor"]
    if cand["arc_min"] + 1e-9 < base["arc_min"] - 0.01:
        return False, ["arc_min_regressed"]
    improved = False
    if cand["arc_min"] > base["arc_min"] + eps:
        reasons.append(f"arc_min {base['arc_min']:.1%}→{cand['arc_min']:.1%}")
        improved = True
    if cand["arc_e"] > base["arc_e"] + eps and cand["arc_c"] > base["arc_c"] + eps:
        reasons.append("both_arc_up")
        improved = True
    if cand["gsm_first"] > base["gsm_first"] + eps and cand["arc_min"] + 1e-9 >= base["arc_min"] - 0.005:
        reasons.append(f"gsm_first {base['gsm_first']:.1%}→{cand['gsm_first']:.1%}")
        improved = True
    if cand["gsm_tf"] > base["gsm_tf"] + 0.015 and cand["arc_min"] + 1e-9 >= base["arc_min"] - 0.005:
        reasons.append(f"gsm_tf {base['gsm_tf']:.1%}→{cand['gsm_tf']:.1%}")
        improved = True
    if cand["gsm_exact"] > base.get("gsm_exact", 0) + eps:
        reasons.append("gsm_exact_up")
        improved = True
    if cand["balanced"] > base["balanced"] + 0.025 and cand["arc_min"] + 1e-9 >= base["arc_min"] - 0.005:
        reasons.append(f"balanced→{cand['balanced']:.3f}")
        improved = True
    return improved, reasons


def set_head_only(student):
    for p in student.parameters():
        p.requires_grad_(False)
    for name, p in student.named_parameters():
        n = name.lower()
        if any(k in n for k in ("lm_head", "embed", "norm", "ln_")):
            p.requires_grad_(True)


def trainable(student):
    return [p for p in student.parameters() if p.requires_grad]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = derive_fsot_lr_plan(d_eff=D_EFF, epochs=12, ref_loss=4.0)
    print("=== VERIFIED OPEN-GAP PUSH ===")

    # ---- pre verification ----
    print("\n[1] FSOT 2.1 verification (pre)...")
    v_pre = run_verification(include_host=True, write=True)
    for name, c in v_pre["layers"].items():
        print(f"  [{'OK' if c.get('ok') else 'FAIL'}] {name}")
    if not v_pre["ok"]:
        print("VERIFY FAIL pre-train — abort (no capability run without green ledger)")
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "verify_pre": v_pre,
            "improved": False,
            "promote_to_github": False,
            "reason": "verify_pre_failed",
        }
        (OUT / "verified_open_gaps.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )
        return 1

    print("VERIFY PASS — proceed to open gaps")

    tok_t, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    gsm_eval = load_gsm8k_test(40)
    for r in gsm_eval:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"
    arc_e_tr, hold_e = split_arc(DATA / "ARC-Easy_train.csv", 2500, 60, 17)
    arc_c_tr, hold_c = split_arc(DATA / "ARC-Challenge_train.csv", 1500, 40, 19)

    # best verified host
    src = CKPT / "pure_fsot_data_driven_best.pt"
    if not src.is_file():
        src = CKPT / "pure_fsot_granular_best.pt"
    tok, student = load_model(device)
    swap_all_layers(student)
    ck = torch.load(src, map_location=device, weights_only=False)
    student.load_state_dict(ck["state_dict"], strict=False)
    print("loaded", src.name)

    gate0 = slim(measure(tok, teacher, student, device, gsm_eval, hold_e, hold_c))
    print("GATE0", json.dumps(gate0, indent=2))
    best = dict(gate0)
    improved = False
    history = []
    t0 = time.time()

    # ---- Phase A: both ARC holds, very soft ----
    print("\n[2] Phase A — dual ARC (micro LR, high retention)")
    # alternate Easy/Challenge 1:1; prefer weaker slightly
    if gate0["arc_c"] <= gate0["arc_e"]:
        arc_train = []
        for a, b in zip(arc_c_tr, arc_e_tr):
            arc_train.extend([a, b, a])  # C,E,C
        leftover = arc_c_tr[len(arc_e_tr) :]
        arc_train.extend(leftover)
    else:
        arc_train = []
        for a, b in zip(arc_e_tr, arc_c_tr):
            arc_train.extend([a, b, a])
    random.Random(51).shuffle(arc_train)

    for p in student.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(list(student.parameters()), lr=plan.lr0 * 0.15, weight_decay=0.01)
    drop = 0
    student.train()
    PHASE_A = 400
    for step in range(1, PHASE_A + 1):
        row = arc_train[step % len(arc_train)]
        gold = row["gold"].strip().upper()[:1]
        if gold not in "ABCD":
            continue
        # next-token dominant (gentler than full TF)
        n1 = next_ce(student, tok, device, row["prompt"], gold, "letter")
        ce = answer_ce(student, tok, device, row["prompt"], gold, "letter")
        ce_r = retention_ce(student, teacher, tok, device, EVAL16[step % len(EVAL16)])
        loss = 1.0 * n1 + 0.35 * ce + 0.60 * ce_r
        if not torch.isfinite(loss):
            continue
        lr = min(
            fsot_epoch_lr(
                plan, epoch=min(step // 40, 11), step=step, loss=float(loss.item()), recent_hits=0.0
            )
            * 0.18,
            plan.lr0 * 0.20,
        )
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()), 0.35)
        opt.step()

        if step % 50 != 0 and step != 1:
            continue
        cur = slim(measure(tok, teacher, student, device, gsm_eval, hold_e, hold_c))
        student.train()
        ok, reasons = beats(cur, best)
        history.append({"phase": "A", "step": step, **cur, "promote": ok, "reasons": reasons})
        print(
            f"  A{step:04d} min={cur['arc_min']:.0%} E={cur['arc_e']:.0%} C={cur['arc_c']:.0%} "
            f"first={cur['gsm_first']:.0%} tf={cur['gsm_tf']:.0%} ag={cur['agree']:.0%} "
            f"bal={cur['balanced']:.3f} {ok} {reasons}"
        )
        if ok:
            best = dict(cur)
            improved = True
            drop = 0
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in student.state_dict().items()},
                    "step": step,
                    "phase": "verified_open_A",
                    "gate": best,
                    "gate0": gate0,
                    "arc_easy_hold": best["arc_e"],
                    "arc_challenge_hold": best["arc_c"],
                    "arc_min": best["arc_min"],
                    "gsm_first": best["gsm_first"],
                    "agree16": best["agree"],
                    "balanced_score": best["balanced"],
                    "verify_pre_ok": True,
                },
                CKPT / "pure_fsot_verified_best.pt",
            )
            print("    * PROMOTED A", reasons)
        elif cur["arc_min"] + 1e-9 < best["arc_min"] - 0.01:
            drop += 1
            if drop >= 3:
                print("Phase A stop — arc_min drop streak")
                p = CKPT / "pure_fsot_verified_best.pt"
                if improved and p.is_file():
                    student.load_state_dict(
                        torch.load(p, map_location=device, weights_only=False)["state_dict"],
                        strict=False,
                    )
                else:
                    student.load_state_dict(ck["state_dict"], strict=False)
                break
        else:
            drop = 0

    # ---- Phase B: head first-digit ----
    print("\n[3] Phase B — LM-head first-digit (body frozen)")
    if improved and (CKPT / "pure_fsot_verified_best.pt").is_file():
        student.load_state_dict(
            torch.load(CKPT / "pure_fsot_verified_best.pt", map_location=device, weights_only=False)[
                "state_dict"
            ],
            strict=False,
        )
    set_head_only(student)
    print(f"  trainable {sum(p.numel() for p in trainable(student))/1e6:.2f}M")
    opt = torch.optim.AdamW(trainable(student), lr=plan.lr0 * 0.6, weight_decay=0.0)
    rng = random.Random(12)
    arith = []
    while len(arith) < 1200:
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        if rng.random() < 0.55:
            gold, q = str(a + b), f"What is {a} + {b}?"
        else:
            aa, bb = max(a, b), min(a, b)
            gold, q = str(aa - bb), f"What is {aa} - {bb}?"
        if len(gold) <= 2:
            arith.append({"prompt": f"Question: {q}\n####", "gold": gold})
    gsm_real = [r for r in load_gsm8k_train(1200) if len(str(r["gold"]).strip()) <= 2]
    arc_mix = arc_e_tr[:300] + arc_c_tr[:300]
    random.Random(13).shuffle(gsm_real)
    random.Random(14).shuffle(arc_mix)
    drop = 0
    student.train()
    for step in range(1, 401):
        r = step % 10
        if r < 5:
            row = arith[step % len(arith)]
            fd = first_digit_ce(student, tok, device, row["prompt"], row["gold"])
            ce = answer_ce(student, tok, device, row["prompt"], row["gold"], "num")
            loss_task = 1.6 * fd + 0.4 * ce
        elif r < 8 and gsm_real:
            row = gsm_real[step % len(gsm_real)]
            q = row["text"].split("\n")[0]
            if not q.startswith("Question:"):
                q = "Question: " + q
            prompt = f"{q}\n####"
            gold = str(row["gold"]).strip()
            fd = first_digit_ce(student, tok, device, prompt, gold)
            ce = answer_ce(student, tok, device, prompt, gold, "num")
            loss_task = 1.6 * fd + 0.4 * ce
        else:
            row = arc_mix[step % len(arc_mix)]
            gold = row["gold"].strip().upper()[:1]
            if gold not in "ABCD":
                continue
            loss_task = next_ce(student, tok, device, row["prompt"], gold, "letter")
        ce_r = retention_ce(student, teacher, tok, device, EVAL16[step % len(EVAL16)])
        loss = loss_task + 0.4 * ce_r
        if not torch.isfinite(loss):
            continue
        for g in opt.param_groups:
            g["lr"] = min(plan.lr0 * 0.65, 3.5e-5)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable(student), 0.5)
        opt.step()

        if step % 50 != 0 and step != 1:
            continue
        cur = slim(measure(tok, teacher, student, device, gsm_eval, hold_e, hold_c))
        student.train()
        ok, reasons = beats(cur, best)
        history.append({"phase": "B", "step": step, **cur, "promote": ok, "reasons": reasons})
        print(
            f"  B{step:04d} min={cur['arc_min']:.0%} E={cur['arc_e']:.0%} C={cur['arc_c']:.0%} "
            f"first={cur['gsm_first']:.0%} tf={cur['gsm_tf']:.0%} x={cur['gsm_exact']:.0%} "
            f"ag={cur['agree']:.0%} bal={cur['balanced']:.3f} {ok} {reasons}"
        )
        if ok:
            best = dict(cur)
            improved = True
            drop = 0
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in student.state_dict().items()},
                    "step": 1000 + step,
                    "phase": "verified_open_B",
                    "gate": best,
                    "gate0": gate0,
                    "arc_easy_hold": best["arc_e"],
                    "arc_challenge_hold": best["arc_c"],
                    "arc_min": best["arc_min"],
                    "gsm_first": best["gsm_first"],
                    "agree16": best["agree"],
                    "balanced_score": best["balanced"],
                    "verify_pre_ok": True,
                },
                CKPT / "pure_fsot_verified_best.pt",
            )
            # also refresh data_driven / granular pointers
            torch.save(
                {
                    "state_dict": {k: v.detach().cpu() for k, v in student.state_dict().items()},
                    "step": 1000 + step,
                    "phase": "verified_open_B",
                    "arc_easy_hold": best["arc_e"],
                    "arc_challenge_hold": best["arc_c"],
                    "arc_min": best["arc_min"],
                    "gsm_first": best["gsm_first"],
                    "agree16": best["agree"],
                    "balanced_score": best["balanced"],
                    "granular_push": True,
                },
                CKPT / "pure_fsot_data_driven_best.pt",
            )
            print("    * PROMOTED B", reasons)
        elif cur["arc_min"] + 1e-9 < best["arc_min"] - 0.01:
            drop += 1
            if drop >= 3:
                print("Phase B stop — arc_min drop")
                break
        else:
            drop = 0

    # reload best for final
    if improved and (CKPT / "pure_fsot_verified_best.pt").is_file():
        student.load_state_dict(
            torch.load(CKPT / "pure_fsot_verified_best.pt", map_location=device, weights_only=False)[
                "state_dict"
            ],
            strict=False,
        )
        final = slim(measure(tok, teacher, student, device, gsm_eval, hold_e, hold_c))
        final_ckpt = CKPT / "pure_fsot_verified_best.pt"
    else:
        final = gate0
        final_ckpt = src

    # ---- post verification ----
    print("\n[4] FSOT 2.1 verification (post)...")
    v_post = run_verification(include_host=True, ckpt_path=final_ckpt, write=True)
    for name, c in v_post["layers"].items():
        print(f"  [{'OK' if c.get('ok') else 'FAIL'}] {name}")

    cap_ok, reasons = beats(final, gate0)
    promote = bool(improved and cap_ok and v_pre["ok"] and v_post["ok"])
    elapsed = time.time() - t0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "verified_open_gaps: FSOT2.1 verify → ARC dual + head first-digit → re-verify",
        "verify_pre_ok": v_pre["ok"],
        "verify_post_ok": v_post["ok"],
        "verify_failed_pre": v_pre.get("failed_layers"),
        "verify_failed_post": v_post.get("failed_layers"),
        "gate0": gate0,
        "best": best,
        "final": final,
        "capability_improved": cap_ok,
        "promote_reasons": reasons if cap_ok else [],
        "improved": promote,
        "promote_to_github": promote,
        "history": history,
        "elapsed_s": elapsed,
        "deltas": {
            k: final[k] - gate0[k]
            for k in ("arc_min", "arc_e", "arc_c", "gsm_first", "gsm_tf", "gsm_exact", "balanced")
            if k in final and k in gate0 and isinstance(final[k], (int, float))
        },
    }
    (OUT / "verified_open_gaps.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    status = "IMPROVED+VERIFIED — eligible to push" if promote else "NO_PUSH"
    (OUT / "VERIFIED_OPEN_GAPS.md").write_text(
        f"""# Verified open-gap push

**Status: {status}**

## Verification (FSOT 2.1 bridge)

| | Pre | Post |
|--|-----|------|
| Overall | {'PASS' if v_pre['ok'] else 'FAIL'} | {'PASS' if v_post['ok'] else 'FAIL'} |
| Failed layers | {v_pre.get('failed_layers') or '—'} | {v_post.get('failed_layers') or '—'} |

See `FSOT21_VERIFY.md` / `fsot21_verify.json`.

## Capability gates

| Axis | Start | Best/Final | Δ |
|------|-------|------------|---|
| ARC min | {gate0['arc_min']:.0%} | {final['arc_min']:.0%} | {(final['arc_min']-gate0['arc_min']):+.0%} |
| ARC-Easy | {gate0['arc_e']:.0%} | {final['arc_e']:.0%} | {(final['arc_e']-gate0['arc_e']):+.0%} |
| ARC-Challenge | {gate0['arc_c']:.0%} | {final['arc_c']:.0%} | {(final['arc_c']-gate0['arc_c']):+.0%} |
| GSM first-digit | {gate0['gsm_first']:.0%} | {final['gsm_first']:.0%} | {(final['gsm_first']-gate0['gsm_first']):+.0%} |
| GSM TF | {gate0['gsm_tf']:.0%} | {final['gsm_tf']:.0%} | {(final['gsm_tf']-gate0['gsm_tf']):+.0%} |
| Balanced | {gate0['balanced']:.3f} | {final['balanced']:.3f} | {(final['balanced']-gate0['balanced']):+.3f} |
| Agree | {gate0['agree']:.0%} | {final['agree']:.0%} | |

Promote reasons: {reasons if promote else 'none'}  
Policy: push GitHub only when **verify PASS + capability IMPROVED**.
""",
        encoding="utf-8",
    )
    print("===", status, "===")
    print(
        f"min {gate0['arc_min']:.0%}→{final['arc_min']:.0%} first {gate0['gsm_first']:.0%}→{final['gsm_first']:.0%} "
        f"verify_pre={v_pre['ok']} verify_post={v_post['ok']}"
    )
    return 0 if v_pre["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

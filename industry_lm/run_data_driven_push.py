#!/usr/bin/env python3
"""
Data-driven push v2 — gate-only promote; never push without improvement.

Phase A — ARC bottleneck only (lift min(Easy, Challenge) without GSM CE).
Phase B — If ARC min holds, LM-head (+norm) first-digit CE only (don't touch body).

Gates: arc_min, gsm_first, gsm_tf, balanced. Agree floor 90%.
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


def split_arc(path: Path, train_n: int, hold_n: int, seed: int):
    all_rows = load_arc_train(path, None)
    rng = random.Random(seed)
    idx = list(range(len(all_rows)))
    rng.shuffle(idx)
    hold = [all_rows[i] for i in idx[:hold_n]]
    train = [all_rows[i] for i in idx[hold_n : hold_n + train_n]]
    return train, hold


def gold_ids(tok, gold: str, kind: str = "num") -> list[int]:
    gold = str(gold).strip()
    cands = [f" {gold}", gold] if kind == "letter" else [f" {gold}", gold, f" {gold}\n"]
    for c in cands:
        ids = tok.encode(c, add_special_tokens=False)
        if ids:
            return ids
    return []


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
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = full[:, 1:].contiguous()
    n = shift_labels.size(1)
    start = max(prompt_len - 1, 0)
    if start >= n:
        return torch.tensor(0.0, device=device)
    mask = torch.zeros(1, n, device=device)
    mask[:, start:] = 1.0
    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(1, n)
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


def next_ce(student, tok, device, prompt, gold, kind="num"):
    gids = gold_ids(tok, gold, kind=kind)
    if not gids:
        return torch.tensor(0.0, device=device, requires_grad=True)
    pe = tok(prompt, return_tensors="pt", truncation=True, max_length=400).to(device)
    return F.cross_entropy(
        student(**pe).logits[0, -1].float().unsqueeze(0),
        torch.tensor([gids[0]], device=device),
    )


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
        "arc_e", "arc_c", "arc_min", "arc_mean", "gsm_first", "gsm_tf",
        "gsm_exact", "gsm_const", "mode", "mode_frac", "agree", "balanced",
    )
    return {k: g[k] for k in keys if k in g}


def gate_pack(m):
    e, c = float(m["arc_e"]), float(m["arc_c"])
    first, tf, ag = float(m["gsm_first"]), float(m["gsm_tf"]), float(m["agree"])
    mn = min(e, c)
    mean = 0.5 * (e + c)
    bal = 2.0 * mn + 1.5 * mean + 1.2 * first + 0.8 * tf + 0.4 * ag
    return {**m, "arc_min": mn, "arc_mean": mean, "balanced": bal}


def measure(tok, teacher, student, device, gsm_eval, hold_e, hold_c):
    student.eval()
    g, _ = eval_gsm_granular(tok, student, device, gsm_eval, arm="fsot")
    ae, _ = eval_arc_granular(tok, student, device, hold_e, arm="fsot")
    ac, _ = eval_arc_granular(tok, student, device, hold_c, arm="fsot")
    ag = agree_n(tok, teacher, student, device, EVAL16)
    return gate_pack({
        "arc_e": ae["exact"] or 0.0,
        "arc_c": ac["exact"] or 0.0,
        "gsm_first": g["first_digit"] or 0.0,
        "gsm_tf": g["tf_token_acc"] or 0.0,
        "gsm_exact": g["exact"] or 0.0,
        "gsm_const": g["constrained_exact"] or 0.0,
        "mode": g.get("mode_pred"),
        "mode_frac": g.get("mode_frac") or 0.0,
        "agree": ag,
    })


def beats_gate(cand, base, eps=1e-4):
    reasons = []
    if cand["agree"] < 0.90:
        return False, ["agree_floor"]
    if cand["arc_min"] + 1e-9 < base["arc_min"] - 0.015:
        return False, ["arc_min_regressed"]
    if cand["gsm_first"] + 1e-9 < base["gsm_first"] - 0.05:
        return False, ["gsm_first_regressed"]
    improved = False
    if cand["arc_min"] > base["arc_min"] + eps:
        reasons.append(f"arc_min {base['arc_min']:.1%}→{cand['arc_min']:.1%}")
        improved = True
    if cand["gsm_first"] > base["gsm_first"] + eps and cand["arc_min"] + 1e-9 >= base["arc_min"] - 0.005:
        reasons.append(f"gsm_first {base['gsm_first']:.1%}→{cand['gsm_first']:.1%}")
        improved = True
    if cand["gsm_tf"] > base["gsm_tf"] + 0.02 and cand["arc_min"] + 1e-9 >= base["arc_min"] - 0.005:
        reasons.append(f"gsm_tf {base['gsm_tf']:.1%}→{cand['gsm_tf']:.1%}")
        improved = True
    if cand["gsm_exact"] > base.get("gsm_exact", 0) + eps:
        reasons.append(f"gsm_exact→{cand['gsm_exact']:.1%}")
        improved = True
    if (
        cand["arc_e"] > base["arc_e"] + eps
        and cand["arc_c"] > base["arc_c"] + eps
    ):
        reasons.append("both_arc_up")
        improved = True
    if cand["balanced"] > base["balanced"] + 0.03 and cand["arc_min"] + 1e-9 >= base["arc_min"] - 0.005:
        reasons.append(f"balanced {base['balanced']:.3f}→{cand['balanced']:.3f}")
        improved = True
    return improved, reasons


def set_trainable(student, mode: str):
    """mode: full | head — head = lm_head + norms + embeddings only."""
    for p in student.parameters():
        p.requires_grad_(False)
    if mode == "full":
        for p in student.parameters():
            p.requires_grad_(True)
        return
    for name, p in student.named_parameters():
        n = name.lower()
        if any(k in n for k in ("lm_head", "embed", "norm", "ln_")):
            p.requires_grad_(True)


def trainable_params(student):
    return [p for p in student.parameters() if p.requires_grad]


def promote(student, step, sc, gate_base, reasons, phase):
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in student.state_dict().items()},
        "step": step,
        "phase": phase,
        "gate": sc,
        "gate_base": gate_base,
        "arc_easy_hold": sc["arc_e"],
        "arc_challenge_hold": sc["arc_c"],
        "arc_min": sc["arc_min"],
        "gsm_first": sc["gsm_first"],
        "gsm_tf": sc["gsm_tf"],
        "gsm_exact": sc["gsm_exact"],
        "agree16": sc["agree"],
        "balanced_score": sc["balanced"],
        "full_dof": True,
        "D_eff": D_EFF,
        "promote_reasons": reasons,
    }
    torch.save(payload, CKPT / "pure_fsot_data_driven_best.pt")
    torch.save(
        {
            **{k: payload[k] for k in payload if k != "state_dict"},
            "state_dict": payload["state_dict"],
            "granular_push": True,
        },
        CKPT / "pure_fsot_granular_best.pt",
    )


def short_arith(n, seed):
    rng = random.Random(seed)
    rows = []
    while len(rows) < n:
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        if rng.random() < 0.6:
            gold = str(a + b)
            q = f"What is {a} + {b}?"
        else:
            aa, bb = max(a, b), min(a, b)
            gold = str(aa - bb)
            q = f"What is {aa} - {bb}?"
        if len(gold) > 2:
            continue
        rows.append({"prompt": f"Question: {q}\n####", "gold": gold})
    return rows


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = derive_fsot_lr_plan(d_eff=D_EFF, epochs=12, ref_loss=4.0)
    print("=== DATA-DRIVEN PUSH v2 (ARC bottleneck → head first-digit) ===")
    print(f"LR lr0={plan.lr0:.3e}")

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

    # pick host by arc_min then balanced
    scores = []
    for src in [
        CKPT / "pure_fsot_answer_locked_best.pt",
        CKPT / "pure_fsot_granular_best.pt",
        CKPT / "pure_fsot_12x3_best.pt",
    ]:
        if not src.is_file():
            continue
        tok, m = load_model(device)
        swap_all_layers(m)
        ck = torch.load(src, map_location=device, weights_only=False)
        m.load_state_dict(ck["state_dict"], strict=False)
        met = measure(tok, teacher, m, device, gsm_eval, hold_e, hold_c)
        print(
            f"  {src.name}: min={met['arc_min']:.0%} E={met['arc_e']:.0%} "
            f"C={met['arc_c']:.0%} first={met['gsm_first']:.0%} tf={met['gsm_tf']:.0%} "
            f"bal={met['balanced']:.3f}"
        )
        scores.append((met["arc_min"], met["balanced"], src))
        del m
        if device == "cuda":
            torch.cuda.empty_cache()

    scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_src = scores[0][2]
    tok, student = load_model(device)
    swap_all_layers(student)
    ck = torch.load(best_src, map_location=device, weights_only=False)
    student.load_state_dict(ck["state_dict"], strict=False)
    gate0 = measure(tok, teacher, student, device, gsm_eval, hold_e, hold_c)
    gate_base = slim(gate0)
    print("START", best_src.name, "GATE", json.dumps(gate_base, indent=2))

    best_gate = dict(gate_base)
    best_step = 0
    improved_once = False
    history = []
    t0 = time.time()

    # ---------- Phase A: ARC bottleneck ----------
    # Overweight whichever hold is weaker
    weak_is_c = gate_base["arc_c"] <= gate_base["arc_e"]
    if weak_is_c:
        arc_train = arc_c_tr + arc_c_tr + arc_e_tr  # 2:1 challenge
        print("Phase A: overweight Challenge (bottleneck)")
    else:
        arc_train = arc_e_tr + arc_e_tr + arc_c_tr
        print("Phase A: overweight Easy (bottleneck)")
    random.Random(41).shuffle(arc_train)

    set_trainable(student, "full")
    opt = torch.optim.AdamW(trainable_params(student), lr=plan.lr0 * 0.2, weight_decay=0.01)
    drop = 0
    student.train()
    PHASE_A = 600
    EVAL_EVERY = 50

    for step in range(1, PHASE_A + 1):
        row = arc_train[step % len(arc_train)]
        gold = row["gold"].strip().upper()[:1]
        if gold not in "ABCD":
            continue
        # soft: next-token letter + light multi-token, high retention
        ce = answer_ce(student, tok, device, row["prompt"], gold, "letter")
        n1 = next_ce(student, tok, device, row["prompt"], gold, "letter")
        ce_r = retention_ce(student, teacher, tok, device, EVAL16[step % len(EVAL16)])
        loss = 0.6 * ce + 1.0 * n1 + 0.55 * ce_r
        if not torch.isfinite(loss):
            continue
        lr = min(
            fsot_epoch_lr(plan, epoch=min(step // 50, 11), step=step, loss=float(loss.item()), recent_hits=0.0)
            * 0.22,
            plan.lr0 * 0.25,
        )
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params(student), 0.4)
        opt.step()

        if step % EVAL_EVERY != 0 and step != 1:
            continue
        cur = slim(measure(tok, teacher, student, device, gsm_eval, hold_e, hold_c))
        student.train()
        ok, reasons = beats_gate(cur, best_gate)
        history.append({"phase": "A", "step": step, **cur, "promote": ok, "reasons": reasons})
        print(
            f"  A{step:04d} min={cur['arc_min']:.0%} E={cur['arc_e']:.0%} C={cur['arc_c']:.0%} "
            f"first={cur['gsm_first']:.0%} tf={cur['gsm_tf']:.0%} ag={cur['agree']:.0%} "
            f"bal={cur['balanced']:.3f} promote={ok} {reasons}"
        )
        if ok:
            best_gate = dict(cur)
            best_step = step
            improved_once = True
            drop = 0
            promote(student, step, cur, gate_base, reasons, "data_driven_A_arc")
            print(f"    * PROMOTED A {reasons}")
        else:
            if cur["arc_min"] + 1e-9 < best_gate["arc_min"] - 0.015:
                drop += 1
            else:
                drop = 0
            if drop >= 3:
                print("Phase A: arc_min drop streak — restore best / end A")
                p = CKPT / "pure_fsot_data_driven_best.pt"
                if improved_once and p.is_file():
                    ck = torch.load(p, map_location=device, weights_only=False)
                    student.load_state_dict(ck["state_dict"], strict=False)
                else:
                    # restore start host
                    ck = torch.load(best_src, map_location=device, weights_only=False)
                    student.load_state_dict(ck["state_dict"], strict=False)
                break

    # reload best for phase B
    p_best = CKPT / "pure_fsot_data_driven_best.pt"
    if improved_once and p_best.is_file():
        ck = torch.load(p_best, map_location=device, weights_only=False)
        student.load_state_dict(ck["state_dict"], strict=False)
        print("Phase B starts from promoted A")
    else:
        ck = torch.load(best_src, map_location=device, weights_only=False)
        student.load_state_dict(ck["state_dict"], strict=False)
        print("Phase B starts from original host (no A promote)")

    # ---------- Phase B: LM-head first-digit only ----------
    print("Phase B: lm_head/norm first-digit (body frozen)")
    set_trainable(student, "head")
    n_train = sum(p.numel() for p in trainable_params(student))
    print(f"  trainable params: {n_train/1e6:.2f}M")
    opt = torch.optim.AdamW(trainable_params(student), lr=plan.lr0 * 0.5, weight_decay=0.0)
    arith = short_arith(1500, seed=7)
    gsm_real = [r for r in load_gsm8k_train(1500) if len(str(r["gold"]).strip()) <= 2]
    random.Random(9).shuffle(gsm_real)
    # keep a little ARC next-token to hold min
    arc_hold_mix = arc_e_tr[:400] + arc_c_tr[:400]
    random.Random(10).shuffle(arc_hold_mix)

    drop = 0
    PHASE_B = 500
    student.train()
    for step in range(1, PHASE_B + 1):
        r = step % 10
        if r < 6:
            row = arith[step % len(arith)]
            prompt, gold = row["prompt"], row["gold"]
            fd = first_digit_ce(student, tok, device, prompt, gold)
            ce = answer_ce(student, tok, device, prompt, gold, "num")
            loss_task = 1.5 * fd + 0.5 * ce
            task = "digit"
        elif r < 8 and gsm_real:
            row = gsm_real[step % len(gsm_real)]
            q = row["text"].split("\n")[0]
            if not q.startswith("Question:"):
                q = "Question: " + q
            prompt = f"{q}\n####"
            gold = str(row["gold"]).strip()
            fd = first_digit_ce(student, tok, device, prompt, gold)
            ce = answer_ce(student, tok, device, prompt, gold, "num")
            loss_task = 1.5 * fd + 0.4 * ce
            task = "gsm"
        else:
            row = arc_hold_mix[step % len(arc_hold_mix)]
            gold = row["gold"].strip().upper()[:1]
            if gold not in "ABCD":
                continue
            loss_task = next_ce(student, tok, device, row["prompt"], gold, "letter")
            task = "arc_hold"

        ce_r = retention_ce(student, teacher, tok, device, EVAL16[step % len(EVAL16)])
        loss = loss_task + 0.35 * ce_r
        if not torch.isfinite(loss):
            continue
        lr = min(plan.lr0 * 0.55, 3e-5)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params(student), 0.5)
        opt.step()

        if step % EVAL_EVERY != 0 and step != 1:
            continue
        cur = slim(measure(tok, teacher, student, device, gsm_eval, hold_e, hold_c))
        student.train()
        ok, reasons = beats_gate(cur, best_gate)
        history.append({"phase": "B", "step": step, "task": task, **cur, "promote": ok, "reasons": reasons})
        print(
            f"  B{step:04d} min={cur['arc_min']:.0%} E={cur['arc_e']:.0%} C={cur['arc_c']:.0%} "
            f"first={cur['gsm_first']:.0%} tf={cur['gsm_tf']:.0%} x={cur['gsm_exact']:.0%} "
            f"ag={cur['agree']:.0%} bal={cur['balanced']:.3f} promote={ok} {reasons}"
        )
        if ok:
            best_gate = dict(cur)
            best_step = 1000 + step
            improved_once = True
            drop = 0
            promote(student, best_step, cur, gate_base, reasons, "data_driven_B_head")
            print(f"    * PROMOTED B {reasons}")
        else:
            if cur["arc_min"] + 1e-9 < best_gate["arc_min"] - 0.015:
                drop += 1
            else:
                drop = 0
            if drop >= 3:
                print("Phase B: arc_min drop — stop")
                break

    # final from best promoted or base
    if improved_once and p_best.is_file():
        ck = torch.load(p_best, map_location=device, weights_only=False)
        student.load_state_dict(ck["state_dict"], strict=False)
        final = slim(measure(tok, teacher, student, device, gsm_eval, hold_e, hold_c))
    else:
        final = gate_base

    vs, reasons_f = beats_gate(final, gate_base)
    promote_gh = bool(improved_once and vs)
    elapsed = time.time() - t0
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "data_driven_v2: PhaseA ARC bottleneck full; PhaseB head first-digit",
        "start_host": best_src.name,
        "gate_base": gate_base,
        "gate_best": best_gate,
        "final": final,
        "improved": promote_gh,
        "promote_to_github": promote_gh,
        "promote_reasons": reasons_f if promote_gh else [],
        "best_step": best_step,
        "history": history,
        "elapsed_s": elapsed,
        "deltas": {
            k: final[k] - gate_base[k]
            for k in ("arc_min", "arc_e", "arc_c", "gsm_first", "gsm_tf", "gsm_exact", "balanced")
        },
    }
    (OUT / "data_driven_push.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    status = "IMPROVED — eligible to push" if promote_gh else "NO_IMPROVE — do not push"
    (OUT / "DATA_DRIVEN_PUSH.md").write_text(
        f"""# Data-driven push v2

**Status: {status}**

| Axis | Base | Best | Δ |
|------|------|------|---|
| ARC min | {gate_base['arc_min']:.0%} | {best_gate['arc_min']:.0%} | {(best_gate['arc_min']-gate_base['arc_min']):+.0%} |
| ARC-Easy | {gate_base['arc_e']:.0%} | {best_gate['arc_e']:.0%} | {(best_gate['arc_e']-gate_base['arc_e']):+.0%} |
| ARC-Challenge | {gate_base['arc_c']:.0%} | {best_gate['arc_c']:.0%} | {(best_gate['arc_c']-gate_base['arc_c']):+.0%} |
| GSM first-digit | {gate_base['gsm_first']:.0%} | {best_gate['gsm_first']:.0%} | {(best_gate['gsm_first']-gate_base['gsm_first']):+.0%} |
| GSM TF | {gate_base['gsm_tf']:.0%} | {best_gate['gsm_tf']:.0%} | {(best_gate['gsm_tf']-gate_base['gsm_tf']):+.0%} |
| Balanced | {gate_base['balanced']:.3f} | {best_gate['balanced']:.3f} | {(best_gate['balanced']-gate_base['balanced']):+.3f} |

Start: `{best_src.name}` · Best step: {best_step} · Reasons: {reasons_f if promote_gh else 'none'}  
Policy: **push GitHub only when IMPROVED.**
""",
        encoding="utf-8",
    )
    print("===", status, "===")
    print(
        f"min {gate_base['arc_min']:.0%}→{best_gate['arc_min']:.0%} | "
        f"first {gate_base['gsm_first']:.0%}→{best_gate['gsm_first']:.0%} | "
        f"tf {gate_base['gsm_tf']:.0%}→{best_gate['gsm_tf']:.0%} | "
        f"bal {gate_base['balanced']:.3f}→{best_gate['balanced']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

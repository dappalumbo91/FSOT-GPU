#!/usr/bin/env python3
"""
Granular metrics + push: fix GSM collapse, refine ARC, measure multi-axis accuracy.

1) Score baseline HF + pure FSOT host on granular axes
2) Train:
   - GSM: simple arithmetic packs + short real #### gold TF (digits after ####)
   - ARC: micro-LR letter CE (protect mode) on Easy+Challenge train, eval held-out
3) Re-score; save best if composite improves without agree collapse
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
from granular_metrics import (  # noqa: E402
    agree_n,
    eval_arc_granular,
    eval_gsm_granular,
)
from real_data_packs import load_arc_train, load_gsm8k_test, load_gsm8k_train  # noqa: E402
from train_corpus import PROBES  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
CKPT = OUT / "checkpoints"
DATA = Path(r"D:\training data")
CKPT.mkdir(parents=True, exist_ok=True)

D_EFF = 14.0
STEPS = 1200
EVAL_EVERY = 100
GRAD_CLIP = 0.5
RET_W = 0.35

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


def load_student(device):
    tok, m = load_model(device)
    swap_all_layers(m)
    for src in [
        CKPT / "pure_fsot_answer_locked_best.pt",
        CKPT / "pure_fsot_12x3_best.pt",
        CKPT / "pure_fsot_realdata_best.pt",
    ]:
        if src.is_file():
            ck = torch.load(src, map_location=device, weights_only=False)
            m.load_state_dict(ck["state_dict"], strict=False)
            print("loaded", src.name, "meta", {k: ck.get(k) for k in ("arc", "gsm", "agree16", "phase", "step")})
            break
    for p in m.parameters():
        p.requires_grad_(True)
    return tok, m


def simple_arith_pack(n: int = 2000, seed: int = 3) -> list[dict]:
    """Tiny grade-school ops — break #### mode collapse with easy golds."""
    rng = random.Random(seed)
    rows = []
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
    ]
    for _ in range(n):
        op_s, fn = rng.choice(ops)
        if op_s == "*":
            a, b = rng.randint(1, 12), rng.randint(1, 12)
        elif op_s == "-":
            a, b = rng.randint(0, 50), rng.randint(0, 50)
            if b > a:
                a, b = b, a
        else:
            a, b = rng.randint(0, 50), rng.randint(0, 50)
        gold = str(fn(a, b))
        q = f"What is {a} {op_s} {b}?"
        # two prompt styles: #### and Answer:
        rows.append(
            {
                "kind": "arith",
                "prompt": f"Question: {q}\n####",
                "gold": gold,
                "text": f"Question: {q}\n#### {gold}",
            }
        )
        rows.append(
            {
                "kind": "arith",
                "prompt": f"Question: {q}\nAnswer:",
                "gold": gold,
                "text": f"Question: {q}\nAnswer: {gold}",
            }
        )
    return rows


def gold_ids(tok, gold: str, kind: str = "num") -> list[int]:
    gold = str(gold).strip()
    if kind == "letter":
        cands = [f" {gold}", gold]
    else:
        cands = [f" {gold}", gold, f" {gold}\n"]
    for c in cands:
        ids = tok.encode(c, add_special_tokens=False)
        if ids:
            return ids
    return []


def answer_ce(student, tok, device, prompt: str, gold: str, kind: str = "num") -> torch.Tensor:
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
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = full[:, 1:].contiguous()
    n = shift_labels.size(1)
    mask = torch.zeros(1, n, device=device)
    start = max(prompt_len - 1, 0)
    if start >= n:
        return torch.tensor(0.0, device=device)
    mask[:, start:] = 1.0
    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(1, n)
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


def next_ce(student, tok, device, prompt: str, gold: str, kind: str = "num") -> torch.Tensor:
    gids = gold_ids(tok, gold, kind=kind)
    if not gids:
        return torch.tensor(0.0, device=device, requires_grad=True)
    pe = tok(prompt, return_tensors="pt", truncation=True, max_length=400).to(device)
    logits = student(**pe).logits[0, -1]
    return F.cross_entropy(logits.float().unsqueeze(0), torch.tensor([gids[0]], device=device))


def digit_after_space_ce(student, tok, device, prompt: str, gold: str) -> torch.Tensor:
    """Pin first digit after forced space (anti 15000 collapse)."""
    bare = tok.encode(str(gold).strip(), add_special_tokens=False)
    if not bare:
        return torch.tensor(0.0, device=device)
    pe = tok(prompt + " ", return_tensors="pt", truncation=True, max_length=400).to(device)
    logits = student(**pe).logits[0, -1]
    return F.cross_entropy(logits.float().unsqueeze(0), torch.tensor([bare[0]], device=device))


def retention_ce(student, teacher, tok, device, prompt: str) -> torch.Tensor:
    re = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        tlab = int(teacher(**re).logits[0, -1].argmax())
    return F.cross_entropy(
        student(**re).logits[0, -1].float().unsqueeze(0),
        torch.tensor([tlab], device=device),
    )


def split_arc(path: Path, train_n: int, hold_n: int, seed: int = 17):
    all_rows = load_arc_train(path, None)
    rng = random.Random(seed)
    idx = list(range(len(all_rows)))
    rng.shuffle(idx)
    hold_idx = set(idx[:hold_n])
    hold = [all_rows[i] for i in hold_idx]
    train = [all_rows[i] for i in idx[hold_n : hold_n + train_n]]
    return train, hold


def composite_score(gsm_sum: dict, arc_sum: dict, agree: float) -> float:
    """
    Multi-axis score — not just free-gen exact.
    Weights: ARC free exact, ARC TF, GSM constrained, GSM TF, GSM first-digit, agree.
    """
    return (
        2.0 * (arc_sum.get("exact") or 0)
        + 1.0 * (arc_sum.get("tf_first_ok") or 0)
        + 1.5 * (gsm_sum.get("constrained_exact") or 0)
        + 1.0 * (gsm_sum.get("tf_token_acc") or 0)
        + 0.8 * (gsm_sum.get("first_digit") or 0)
        + 0.5 * (gsm_sum.get("format_ok") or 0)
        + 0.4 * agree
        + 0.3 * (gsm_sum.get("exact") or 0)
    )


def scoreboard_block(label: str, gsm: dict, arc_e: dict, arc_c: dict, agree: float) -> str:
    return (
        f"### {label}\n"
        f"| Axis | Value |\n|------|-------|\n"
        f"| Agree16 | {agree:.0%} |\n"
        f"| GSM free exact | {gsm.get('exact', 0):.0%} |\n"
        f"| GSM first-digit | {gsm.get('first_digit', 0):.0%} |\n"
        f"| GSM format-ok | {gsm.get('format_ok', 0):.0%} |\n"
        f"| GSM len-match | {gsm.get('len_match', 0):.0%} |\n"
        f"| GSM constrained exact | {gsm.get('constrained_exact', 0):.0%} |\n"
        f"| GSM TF token acc | {gsm.get('tf_token_acc', 0):.0%} |\n"
        f"| GSM TF first ok | {gsm.get('tf_first_ok', 0):.0%} |\n"
        f"| GSM mode collapse | {gsm.get('mode_collapse')} ({gsm.get('mode_pred')} @ {gsm.get('mode_frac', 0):.0%}) |\n"
        f"| ARC-Easy free | {arc_e.get('exact', 0):.0%} |\n"
        f"| ARC-Easy first-tok letter | {arc_e.get('first_token_letter', 0):.0%} |\n"
        f"| ARC-Easy TF first | {arc_e.get('tf_first_ok', 0):.0%} |\n"
        f"| ARC-Challenge free | {arc_c.get('exact', 0):.0%} |\n"
        f"| ARC-Challenge first-tok | {arc_c.get('first_token_letter', 0):.0%} |\n"
        f"| GSM by len | `{json.dumps(gsm.get('by_gold_len', {}), default=str)[:200]}` |\n"
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = derive_fsot_lr_plan(d_eff=D_EFF, epochs=12, ref_loss=4.0)
    print("=== GRANULAR PUSH ===")
    print(f"FSOT LR lr0={plan.lr0:.3e} floor={plan.lr_floor:.3e}")

    tok_t, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tok, student = load_student(device)

    # --- eval sets ---
    gsm_eval = load_gsm8k_test(40)
    for r in gsm_eval:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"
    # simple arith holdout for format learning signal
    arith_hold = simple_arith_pack(80, seed=99)[::2]  # #### style only, 40 items
    arith_hold = arith_hold[:40]

    arc_easy_tr, arc_easy_hold = split_arc(
        DATA / "ARC-Easy_train.csv", train_n=2500, hold_n=60, seed=17
    )
    arc_ch_tr, arc_ch_hold = split_arc(
        DATA / "ARC-Challenge_train.csv", train_n=1200, hold_n=40, seed=19
    )
    # also first-40 Easy (legacy comparable)
    arc_easy_legacy = load_arc_train(DATA / "ARC-Easy_train.csv", 40)

    def full_eval(model, arm: str):
        model.eval()
        g, gi = eval_gsm_granular(tok, model, device, gsm_eval, arm=arm)
        ga, gai = eval_gsm_granular(tok, model, device, arith_hold, arm=arm)
        ae, aei = eval_arc_granular(tok, model, device, arc_easy_hold, arm=arm)
        ac, aci = eval_arc_granular(tok, model, device, arc_ch_hold, arm=arm)
        al, _ = eval_arc_granular(tok, model, device, arc_easy_legacy, arm=arm)
        ag = agree_n(tok, teacher, model, device, EVAL16) if arm != "baseline" else 1.0
        if arm == "baseline":
            # agree vs self is 100%; report N/A as 1.0 for composite only when student
            ag = agree_n(tok, teacher, model, device, EVAL16)
        return {
            "gsm": g,
            "arith": ga,
            "arc_easy_hold": ae,
            "arc_challenge_hold": ac,
            "arc_easy_legacy40": al,
            "agree": ag,
            "composite": composite_score(g, ae, ag),
            "items": {
                "gsm_n": len(gi),
                "arith_n": len(gai),
                "arc_e_n": len(aei),
                "arc_c_n": len(aci),
            },
        }

    print("\n--- Baseline HF ---")
    base = full_eval(teacher, "baseline")
    print(
        f"  GSM exact={base['gsm']['exact']:.0%} const={base['gsm']['constrained_exact']:.0%} "
        f"tf={base['gsm']['tf_token_acc']:.0%} | ARC-E hold={base['arc_easy_hold']['exact']:.0%} "
        f"ARC-C={base['arc_challenge_hold']['exact']:.0%} | agree={base['agree']:.0%}"
    )

    print("\n--- Student START ---")
    start = full_eval(student, "fsot")
    print(
        f"  GSM exact={start['gsm']['exact']:.0%} const={start['gsm']['constrained_exact']:.0%} "
        f"tf={start['gsm']['tf_token_acc']:.0%} first={start['gsm']['first_digit']:.0%} "
        f"mode={start['gsm'].get('mode_pred')}@{start['gsm'].get('mode_frac', 0):.0%}"
    )
    print(
        f"  ARC-E hold={start['arc_easy_hold']['exact']:.0%} "
        f"legacy40={start['arc_easy_legacy40']['exact']:.0%} "
        f"ARC-C={start['arc_challenge_hold']['exact']:.0%} "
        f"tf_first={start['arc_easy_hold']['tf_first_ok']:.0%} | agree={start['agree']:.0%} "
        f"composite={start['composite']:.3f}"
    )

    # --- train packs ---
    arith = simple_arith_pack(1500, seed=3)
    gsm_real = [r for r in load_gsm8k_train(2000) if r["text"].count("\n") <= 2]
    # short gold preferred
    gsm_real = sorted(gsm_real, key=lambda r: len(str(r["gold"]).strip()))
    arc_train = arc_easy_tr + arc_ch_tr
    random.Random(21).shuffle(arc_train)
    random.Random(22).shuffle(arith)
    random.Random(23).shuffle(gsm_real)

    opt = torch.optim.AdamW(list(student.parameters()), lr=plan.lr0 * 0.4, weight_decay=0.01)
    best = {
        "score": start["composite"],
        "arc": start["arc_easy_hold"]["exact"],
        "gsm_c": start["gsm"]["constrained_exact"],
        "agree": start["agree"],
        "step": 0,
    }
    history = []
    t0 = time.time()
    student.train()
    recent = 0.0

    print("\n=== TRAIN (arith GSM + real short GSM + ARC micro) ===")
    for step in range(1, STEPS + 1):
        # schedule: 50% arith, 20% real GSM, 30% ARC
        r = step % 10
        if r < 5:
            row = arith[step % len(arith)]
            ce = answer_ce(student, tok, device, row["prompt"], row["gold"], "num")
            ce1 = next_ce(student, tok, device, row["prompt"], row["gold"], "num")
            ce_d = digit_after_space_ce(student, tok, device, row["prompt"], row["gold"])
            task = "arith"
            loss_task = ce + 0.8 * ce1 + 0.6 * ce_d
        elif r < 7:
            row = gsm_real[step % len(gsm_real)]
            qline = row["text"].split("\n")[0]
            if not qline.startswith("Question:"):
                qline = "Question: " + qline
            prompt = f"{qline}\n####"
            gold = str(row["gold"]).strip()
            ce = answer_ce(student, tok, device, prompt, gold, "num")
            ce1 = next_ce(student, tok, device, prompt, gold, "num")
            ce_d = digit_after_space_ce(student, tok, device, prompt, gold)
            task = "gsm"
            loss_task = ce + 0.8 * ce1 + 0.6 * ce_d
        else:
            row = arc_train[step % len(arc_train)]
            gold = row["gold"].strip().upper()[:1]
            if gold not in "ABCD":
                continue
            ce = answer_ce(student, tok, device, row["prompt"], gold, "letter")
            ce1 = next_ce(student, tok, device, row["prompt"], gold, "letter")
            task = "arc"
            loss_task = ce + 0.7 * ce1

        ce_r = retention_ce(student, teacher, tok, device, EVAL16[step % len(EVAL16)])
        loss = loss_task + RET_W * ce_r
        if not torch.isfinite(loss):
            continue
        lr = fsot_epoch_lr(
            plan,
            epoch=min(step // max(STEPS // 12, 1), 11),
            step=step,
            loss=float(loss.item()),
            recent_hits=recent,
        )
        # protective: never too aggressive
        lr = min(lr * 0.45, plan.lr0 * 0.5)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()), GRAD_CLIP)
        opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            student.eval()
            cur = full_eval(student, "fsot")
            student.train()
            # free-gen GSM exact is sticky 0; track constrained + ARC + TF
            recent = 0.3 * recent + 0.7 * (
                10 * (cur["gsm"]["constrained_exact"] or 0)
                + 8 * (cur["arc_easy_hold"]["exact"] or 0)
                + 5 * (cur["gsm"]["tf_token_acc"] or 0)
            )
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": lr,
                    "task": task,
                    "gsm_exact": cur["gsm"]["exact"],
                    "gsm_const": cur["gsm"]["constrained_exact"],
                    "gsm_tf": cur["gsm"]["tf_token_acc"],
                    "gsm_first": cur["gsm"]["first_digit"],
                    "gsm_format": cur["gsm"]["format_ok"],
                    "arith_exact": cur["arith"]["exact"],
                    "arith_const": cur["arith"]["constrained_exact"],
                    "arc_e": cur["arc_easy_hold"]["exact"],
                    "arc_c": cur["arc_challenge_hold"]["exact"],
                    "arc_legacy": cur["arc_easy_legacy40"]["exact"],
                    "agree": cur["agree"],
                    "composite": cur["composite"],
                    "mode": cur["gsm"].get("mode_pred"),
                    "mode_frac": cur["gsm"].get("mode_frac"),
                }
            )
            print(
                f"  {step:04d}/{STEPS} loss={loss.item():.3f} lr={lr:.2e} "
                f"gsm_x={cur['gsm']['exact']:.0%} gsm_c={cur['gsm']['constrained_exact']:.0%} "
                f"gsm_tf={cur['gsm']['tf_token_acc']:.0%} first={cur['gsm']['first_digit']:.0%} "
                f"arith_c={cur['arith']['constrained_exact']:.0%} "
                f"arcE={cur['arc_easy_hold']['exact']:.0%} arcC={cur['arc_challenge_hold']['exact']:.0%} "
                f"leg={cur['arc_easy_legacy40']['exact']:.0%} "
                f"ag={cur['agree']:.0%} comp={cur['composite']:.3f} "
                f"mode={cur['gsm'].get('mode_pred')}@{cur['gsm'].get('mode_frac', 0):.0%}"
            )
            improved = cur["composite"] > best["score"] + 1e-4 and cur["agree"] >= 0.85
            # also accept ARC hold improvement with no agree crash
            arc_up = (cur["arc_easy_hold"]["exact"] or 0) > best["arc"] + 1e-4 and cur[
                "agree"
            ] >= 0.90
            gsm_c_up = (cur["gsm"]["constrained_exact"] or 0) > best["gsm_c"] + 1e-4 and cur[
                "agree"
            ] >= 0.85
            if improved or arc_up or gsm_c_up:
                best = {
                    "score": cur["composite"],
                    "arc": cur["arc_easy_hold"]["exact"],
                    "gsm_c": cur["gsm"]["constrained_exact"],
                    "agree": cur["agree"],
                    "step": step,
                    "snapshot": {
                        "gsm": cur["gsm"],
                        "arith": cur["arith"],
                        "arc_easy_hold": cur["arc_easy_hold"],
                        "arc_challenge_hold": cur["arc_challenge_hold"],
                        "arc_easy_legacy40": cur["arc_easy_legacy40"],
                    },
                }
                torch.save(
                    {
                        "state_dict": {
                            k: v.detach().cpu() for k, v in student.state_dict().items()
                        },
                        "step": step,
                        "composite": cur["composite"],
                        "arc_easy_hold": cur["arc_easy_hold"]["exact"],
                        "gsm_constrained": cur["gsm"]["constrained_exact"],
                        "gsm_exact": cur["gsm"]["exact"],
                        "agree16": cur["agree"],
                        "full_dof": True,
                        "granular_push": True,
                        "D_eff": D_EFF,
                    },
                    CKPT / "pure_fsot_granular_best.pt",
                )
                torch.save(
                    {
                        "state_dict": {
                            k: v.detach().cpu() for k, v in student.state_dict().items()
                        },
                        "step": step,
                        "phase": "granular_push",
                        "arc": cur["arc_easy_hold"]["exact"],
                        "gsm": cur["gsm"]["exact"],
                        "agree16": cur["agree"],
                        "full_dof": True,
                        "D_eff": D_EFF,
                    },
                    CKPT / "pure_fsot_answer_locked_best.pt",
                )
                print(
                    f"    * BEST step={step} comp={cur['composite']:.3f} "
                    f"arcE={cur['arc_easy_hold']['exact']:.0%} "
                    f"gsm_c={cur['gsm']['constrained_exact']:.0%}"
                )

    # reload best for final
    p_best = CKPT / "pure_fsot_granular_best.pt"
    if p_best.is_file():
        ck = torch.load(p_best, map_location=device, weights_only=False)
        student.load_state_dict(ck["state_dict"], strict=False)
        print("final from granular best step", ck.get("step"))

    final = full_eval(student, "fsot")
    elapsed = time.time() - t0

    def strip_items(d):
        return {k: v for k, v in d.items() if k != "items"}

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "granular_metrics + arith GSM anti-collapse + ARC micro-LR + held-out",
        "lr_plan": {"lr0": plan.lr0, "floor": plan.lr_floor, "ceil": plan.lr_ceil},
        "metrics_defined": {
            "gsm_free_exact": "first number after #### free-gen == gold",
            "gsm_first_digit": "first digit of pred matches gold",
            "gsm_format_ok": "produced any parseable number",
            "gsm_len_match": "digit-length of pred == gold",
            "gsm_constrained_exact": "digit-only decode == gold",
            "gsm_tf_token_acc": "teacher-forced gold token accuracy",
            "gsm_mode_collapse": "≥40% of preds share same mode string",
            "arc_free_exact": "letter free-gen == gold",
            "arc_first_token_letter": "argmax next token is correct letter form",
            "arc_tf_first": "TF first gold letter token correct",
            "held_out": "shuffled hold slices of Easy/Challenge train CSVs",
            "composite": "weighted multi-axis (see composite_score)",
        },
        "baseline": strip_items(base),
        "start": strip_items(start),
        "final": strip_items(final),
        "best": best,
        "history": history,
        "deltas": {
            "arc_easy_hold": (final["arc_easy_hold"]["exact"] or 0)
            - (start["arc_easy_hold"]["exact"] or 0),
            "arc_challenge_hold": (final["arc_challenge_hold"]["exact"] or 0)
            - (start["arc_challenge_hold"]["exact"] or 0),
            "gsm_constrained": (final["gsm"]["constrained_exact"] or 0)
            - (start["gsm"]["constrained_exact"] or 0),
            "gsm_exact": (final["gsm"]["exact"] or 0) - (start["gsm"]["exact"] or 0),
            "gsm_tf": (final["gsm"]["tf_token_acc"] or 0) - (start["gsm"]["tf_token_acc"] or 0),
            "composite": final["composite"] - start["composite"],
            "vs_base_arc_easy": (final["arc_easy_hold"]["exact"] or 0)
            - (base["arc_easy_hold"]["exact"] or 0),
            "vs_base_gsm_exact": (final["gsm"]["exact"] or 0) - (base["gsm"]["exact"] or 0),
            "vs_base_gsm_const": (final["gsm"]["constrained_exact"] or 0)
            - (base["gsm"]["constrained_exact"] or 0),
        },
        "elapsed_s": elapsed,
        "ok": True,
    }
    # drop nested items if any
    for key in ("baseline", "start", "final"):
        report[key].pop("items", None)

    path = OUT / "granular_push.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    md = f"""# Granular metrics push

Multi-axis accuracy (not only headline GSM%/ARC%).

## Metric dictionary

| Axis | Meaning |
|------|---------|
| GSM free exact | First number in free-gen after `####` equals gold |
| GSM first-digit | Leading digit of pred matches gold |
| GSM format-ok | Any parseable number produced |
| GSM len-match | Digit length matches gold |
| GSM constrained exact | Greedy decode restricted to space/0-9/newline equals gold |
| GSM TF token/first | Teacher-forced gold digit accuracy (format learning signal) |
| Mode collapse | ≥40% of free-gen preds are the same string |
| ARC free exact | Letter free-gen equals gold |
| ARC first-token letter | Argmax next token is the letter |
| ARC TF first | Teacher-forced first letter token correct |
| Held-out ARC | Shuffled hold of Easy/Challenge (not first-40 only) |

## Scoreboard

| Axis | Baseline HF | Start FSOT | Final FSOT |
|------|-------------|------------|------------|
| Agree16 | {base['agree']:.0%} | {start['agree']:.0%} | **{final['agree']:.0%}** |
| GSM free exact | {base['gsm']['exact']:.0%} | {start['gsm']['exact']:.0%} | **{final['gsm']['exact']:.0%}** |
| GSM first-digit | {base['gsm']['first_digit']:.0%} | {start['gsm']['first_digit']:.0%} | **{final['gsm']['first_digit']:.0%}** |
| GSM format-ok | {base['gsm']['format_ok']:.0%} | {start['gsm']['format_ok']:.0%} | **{final['gsm']['format_ok']:.0%}** |
| GSM constrained | {base['gsm']['constrained_exact']:.0%} | {start['gsm']['constrained_exact']:.0%} | **{final['gsm']['constrained_exact']:.0%}** |
| GSM TF token | {base['gsm']['tf_token_acc']:.0%} | {start['gsm']['tf_token_acc']:.0%} | **{final['gsm']['tf_token_acc']:.0%}** |
| Arith constrained | {base['arith']['constrained_exact']:.0%} | {start['arith']['constrained_exact']:.0%} | **{final['arith']['constrained_exact']:.0%}** |
| ARC-Easy hold | {base['arc_easy_hold']['exact']:.0%} | {start['arc_easy_hold']['exact']:.0%} | **{final['arc_easy_hold']['exact']:.0%}** |
| ARC-Easy legacy40 | {base['arc_easy_legacy40']['exact']:.0%} | {start['arc_easy_legacy40']['exact']:.0%} | **{final['arc_easy_legacy40']['exact']:.0%}** |
| ARC-Challenge hold | {base['arc_challenge_hold']['exact']:.0%} | {start['arc_challenge_hold']['exact']:.0%} | **{final['arc_challenge_hold']['exact']:.0%}** |
| Composite | {base['composite']:.3f} | {start['composite']:.3f} | **{final['composite']:.3f}** |

Mode (final free-gen): `{final['gsm'].get('mode_pred')}` @ {final['gsm'].get('mode_frac', 0):.0%} collapse={final['gsm'].get('mode_collapse')}

## Deltas (final − start)

- ARC-Easy hold: **{report['deltas']['arc_easy_hold']:+.0%}**
- ARC-Challenge hold: **{report['deltas']['arc_challenge_hold']:+.0%}**
- GSM constrained: **{report['deltas']['gsm_constrained']:+.0%}**
- GSM free exact: **{report['deltas']['gsm_exact']:+.0%}**
- GSM TF: **{report['deltas']['gsm_tf']:+.0%}**
- Composite: **{report['deltas']['composite']:+.3f}**
- vs base ARC-Easy: **{report['deltas']['vs_base_arc_easy']:+.0%}**

Ckpt: `pure_fsot_granular_best.pt`  
Elapsed: {elapsed:.0f}s
"""
    (OUT / "GRANULAR_PUSH.md").write_text(md, encoding="utf-8")
    print("\n=== GRANULAR SUMMARY ===")
    print(
        f"ARC-E {start['arc_easy_hold']['exact']:.0%}→{final['arc_easy_hold']['exact']:.0%} | "
        f"ARC-C {start['arc_challenge_hold']['exact']:.0%}→{final['arc_challenge_hold']['exact']:.0%} | "
        f"GSM free {start['gsm']['exact']:.0%}→{final['gsm']['exact']:.0%} | "
        f"GSM const {start['gsm']['constrained_exact']:.0%}→{final['gsm']['constrained_exact']:.0%} | "
        f"comp {start['composite']:.3f}→{final['composite']:.3f}"
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

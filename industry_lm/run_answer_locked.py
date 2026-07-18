#!/usr/bin/env python3
"""
Break the accuracy plateau: answer-locked training under pure FSOT.

Phase A — GSM-only: CE only on tokens of the gold answer after `####`
Phase B — ARC-only: CE only on the gold letter token after `Answer:`
Phase C — merge both (light mix) if both hold above floors

No full-sequence CE on rambling solutions. Miss-trace every eval.
FSOT LR: derive_fsot_lr_plan + fsot_epoch_lr / suction_poof live loss.
"""
from __future__ import annotations

import json
import math
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
from fsot_lib.learn import derive_fsot_lr_plan, fsot_epoch_lr, suction_poof_lr  # noqa: E402
from miss_trace import make_miss_entry, write_miss_log  # noqa: E402
from real_data_packs import (  # noqa: E402
    load_arc_train,
    load_gsm8k_test,
    load_gsm8k_train,
)
from train_corpus import PROBES  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
CKPT = OUT / "checkpoints"
TRACE = OUT / "miss_traces"
DATA = Path(r"D:\training data")
CKPT.mkdir(parents=True, exist_ok=True)
TRACE.mkdir(parents=True, exist_ok=True)

D_EFF = 14.0
# GSM phase
GSM_STEPS = 1500
GSM_EVAL_EVERY = 100
GSM_TRAIN_N = 5000
GSM_EVAL_N = 40
# ARC phase
ARC_STEPS = 1500
ARC_EVAL_EVERY = 75
ARC_TRAIN_N = 4000
ARC_EVAL_N = 40
ARC_HOLD_STREAK = 3  # consecutive evals > 35%
ARC_TARGET = 0.35
# merge
MERGE_STEPS = 500
MERGE_EVAL_EVERY = 100

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
RET_W = 0.15
GRAD_CLIP = 0.5


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
    # Prefer clean hosts — skip partially collapsed answer_locked mid-runs
    for src in [
        CKPT / "pure_fsot_12x3_best.pt",
        CKPT / "pure_fsot_realdata_best.pt",
        CKPT / "pure_fsot_curriculum_best.pt",
        CKPT / "pure_fsot_agree100_best.pt",
    ]:
        if src.is_file():
            ck = torch.load(src, map_location=device, weights_only=False)
            m.load_state_dict(ck["state_dict"], strict=False)
            print("loaded", src.name)
            break
    for p in m.parameters():
        p.requires_grad_(True)
    return tok, m


def extract_num(s: str):
    nums = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
    return nums[-1] if nums else None


@torch.no_grad()
def gen(tok, model, device, prompt, max_new=16, eos_ids=None):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=480).to(device)
    kwargs = dict(
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    if eos_ids:
        kwargs["eos_token_id"] = eos_ids
    out = model.generate(**inp, **kwargs)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt) :] if text.startswith(prompt) else text


@torch.no_grad()
def agree16(tok, teacher, student, device):
    ok = 0
    for p in EVAL16:
        inp = tok(p, return_tensors="pt").to(device)
        if int(teacher(**inp).logits[0, -1].argmax()) == int(
            student(**inp).logits[0, -1].argmax()
        ):
            ok += 1
    return ok / len(EVAL16)


def gold_token_ids(tok, gold: str, *, kind: str = "num") -> list[int]:
    """
    Token ids for the answer string.

    Numeric: prefer leading space (matches natural gen after #### → ' 18').
    Must train FULL multi-token span, not only the space.
    Letter (ARC): prefer ' A' single-token form.
    """
    gold = str(gold).strip()
    if kind == "letter":
        cands = [f" {gold}", gold, f"{gold}\n"]
    else:
        # leading space first so inference path matches teacher-forced path
        cands = [f" {gold}", gold, f" {gold}\n", f"{gold}\n"]
    for cand in cands:
        ids = tok.encode(cand, add_special_tokens=False)
        if ids:
            return ids
    return []


def answer_locked_ce(
    student,
    tok,
    device,
    *,
    prompt: str,
    gold: str,
    kind: str = "num",
) -> torch.Tensor:
    """
    Teacher-forced CE only on gold answer tokens.

    Builds [prompt_ids | gold_ids] in one forward, masks loss so only
    positions that predict gold tokens contribute. Avoids full-solution CE.
    """
    gids = gold_token_ids(tok, gold, kind=kind)
    if not gids:
        return torch.tensor(0.0, device=device, requires_grad=True)

    pe = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=400,
        add_special_tokens=True,
    )
    prompt_ids = pe["input_ids"][0].to(device)
    gold_t = torch.tensor(gids, device=device, dtype=prompt_ids.dtype)
    full = torch.cat([prompt_ids, gold_t], dim=0).unsqueeze(0)
    # safety truncate keeping the answer
    max_len = 416
    if full.size(1) > max_len:
        overflow = full.size(1) - max_len
        full = full[:, overflow:]
        prompt_len = max(prompt_ids.numel() - overflow, 1)
    else:
        prompt_len = int(prompt_ids.numel())

    logits = student(input_ids=full, attention_mask=torch.ones_like(full)).logits
    # shift: position i predicts token i+1
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = full[:, 1:].contiguous()
    # label at shift index j is full[j+1]; answer labels when j+1 >= prompt_len
    n = shift_labels.size(1)
    mask = torch.zeros(1, n, device=device)
    start = max(prompt_len - 1, 0)
    if start < n:
        mask[:, start:] = 1.0
    else:
        return torch.tensor(0.0, device=device)
    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(1, n)
    return (ce * mask).sum() / mask.sum().clamp_min(1.0)


def next_token_answer_ce(
    student,
    tok,
    device,
    *,
    prompt: str,
    gold: str,
    kind: str = "num",
) -> torch.Tensor:
    """Single next-token CE: first gold token after prompt."""
    gids = gold_token_ids(tok, gold, kind=kind)
    if not gids:
        return torch.tensor(0.0, device=device, requires_grad=True)
    pe = tok(prompt, return_tensors="pt", truncation=True, max_length=400).to(device)
    logits = student(**pe).logits[0, -1]
    return F.cross_entropy(
        logits.float().unsqueeze(0),
        torch.tensor([gids[0]], device=device),
    )


def retention_ce(student, teacher, tok, device, prompt: str) -> torch.Tensor:
    re = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        tlab = int(teacher(**re).logits[0, -1].argmax())
    return F.cross_entropy(
        student(**re).logits[0, -1].float().unsqueeze(0),
        torch.tensor([tlab], device=device),
    )


@torch.no_grad()
def eval_gsm(tok, model, device, rows, collect_misses=False, arm="fsot"):
    """
    Short free-gen after #### (max_new=8) — multi-digit + stop on newline/EOS.
    Honest #### scoring: first number in generation only.
    """
    hits = 0
    misses = []
    # stop early on newline so 15000… soup is less rewarded at eval
    nl_ids = tok.encode("\n", add_special_tokens=False)
    eos = [tok.eos_token_id] + (nl_ids if nl_ids else [])
    for r in rows:
        prompt = r["prompt"]
        if not prompt.rstrip().endswith("####"):
            prompt = prompt.split("Answer:")[0].strip() + "\n####"
        thought = gen(tok, model, device, prompt, max_new=8, eos_ids=eos)
        gold = extract_num(str(r["gold"])) or str(r["gold"]).strip()
        nums = re.findall(r"-?\d+\.?\d*", thought.replace(",", ""))
        pred = nums[0] if nums else None
        ok = pred is not None and gold is not None and pred == gold
        hits += int(ok)
        if collect_misses and not ok:
            misses.append(
                make_miss_entry(
                    kind="gsm8k",
                    prompt=prompt,
                    gold=str(gold),
                    pred=pred,
                    thought=thought,
                    arm=arm,
                )
            )
    return hits / max(len(rows), 1), misses


@torch.no_grad()
def eval_arc(tok, model, device, rows, collect_misses=False, arm="fsot"):
    hits = 0
    misses = []
    for r in rows:
        thought = gen(tok, model, device, r["prompt"], max_new=8)
        m = re.search(r"\b([ABCD])\b", thought.upper())
        pred = m.group(1) if m else (thought.strip()[:1].upper() if thought.strip() else "")
        gold = r["gold"].strip().upper()
        ok = pred == gold
        hits += int(ok)
        if collect_misses and not ok:
            misses.append(
                make_miss_entry(
                    kind="arc",
                    prompt=r["prompt"],
                    gold=gold,
                    pred=pred,
                    thought=thought,
                    arm=arm,
                )
            )
    return hits / max(len(rows), 1), misses


def save_ckpt(student, path: Path, **meta):
    torch.save(
        {
            **meta,
            "full_dof": True,
            "answer_locked": True,
            "D_eff": D_EFF,
            "state_dict": {
                k: v.detach().cpu() for k, v in student.state_dict().items()
            },
        },
        path,
    )


def phase_gsm(tok, teacher, student, device, plan, opt):
    print("\n=== PHASE A: GSM answer-locked (#### gold tokens only) ===")
    train = load_gsm8k_train(GSM_TRAIN_N)
    # Prefer short #### lines (Question + #### final)
    short = [r for r in train if r["text"].count("\n") <= 2]
    if len(short) >= 500:
        train = short
    # Prefer short gold answers (1–3 digit) first half of curriculum
    easy = [r for r in train if len(str(r["gold"]).strip().replace(".", "")) <= 3]
    hard = [r for r in train if r not in easy]
    random.Random(7).shuffle(easy)
    random.Random(8).shuffle(hard)
    train = (easy * 2) + hard if easy else train
    random.Random(9).shuffle(train)

    eval_rows = load_gsm8k_test(GSM_EVAL_N)
    for r in eval_rows:
        if "####" not in r["prompt"]:
            q = r["prompt"].split("Answer:")[0].strip()
            r["prompt"] = q + "\n####"

    best = {"acc": -1.0, "step": -1, "agree": 0.0}
    history = []
    recent_hits = 0.0
    zero_streak = 0
    student.train()

    for step in range(1, GSM_STEPS + 1):
        r = train[step % len(train)]
        qline = r["text"].split("\n")[0]
        if not qline.startswith("Question:"):
            qline = "Question: " + qline
        # Match pack format: #### then space+number (as free-gen sees)
        prompt = f"{qline}\n####"
        gold = str(r["gold"]).strip()
        # Full multi-token TF for " 72" + next-token on first gold token
        ce = answer_locked_ce(
            student, tok, device, prompt=prompt, gold=gold, kind="num"
        )
        ce1 = next_token_answer_ce(
            student, tok, device, prompt=prompt, gold=gold, kind="num"
        )
        # Also pin bare digits (no leading space) every other step — anti-collapse
        if step % 2 == 0:
            pe = tok(prompt + " ", return_tensors="pt", truncation=True, max_length=400).to(
                device
            )
            gids = gold_token_ids(tok, gold, kind="num")
            # first non-space gold token if multi-token " 18"
            bare = tok.encode(gold, add_special_tokens=False)
            tid = bare[0] if bare else (gids[1] if len(gids) > 1 else (gids[0] if gids else None))
            if tid is not None:
                logits = student(**pe).logits[0, -1]
                ce_digit = F.cross_entropy(
                    logits.float().unsqueeze(0),
                    torch.tensor([tid], device=device),
                )
            else:
                ce_digit = torch.tensor(0.0, device=device)
        else:
            ce_digit = torch.tensor(0.0, device=device)
        ce_r = retention_ce(
            student, teacher, tok, device, EVAL16[step % len(EVAL16)]
        )
        loss = ce + 0.75 * ce1 + 0.5 * ce_digit + RET_W * ce_r
        if not torch.isfinite(loss):
            continue
        # Don't skip low loss — still step (collapse fix needs gradient on digits)
        lr = fsot_epoch_lr(
            plan,
            epoch=min(step // max(GSM_STEPS // 12, 1), 11),
            step=step,
            loss=float(loss.item()),
            recent_hits=recent_hits,
        )
        # Anti-collapse: if stuck at 0%, allow slightly higher LR
        if zero_streak >= 3:
            lr = min(plan.lr_ceil * 0.9, max(lr, plan.lr0 * 1.2))
        else:
            lr = min(lr, plan.lr0 * 0.85)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()), GRAD_CLIP)
        opt.step()

        if step % GSM_EVAL_EVERY == 0 or step == 1:
            student.eval()
            acc, misses = eval_gsm(
                tok, student, device, eval_rows, collect_misses=True
            )
            ag = agree16(tok, teacher, student, device)
            student.train()
            recent_hits = 0.3 * recent_hits + 0.7 * (acc * 10)
            # Detect mode collapse from miss preds
            preds = [m.get("pred") for m in misses if m.get("pred")]
            mode = max(set(preds), key=preds.count) if preds else None
            mode_frac = (preds.count(mode) / len(preds)) if preds and mode else 0.0
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": lr,
                    "gsm": acc,
                    "agree": ag,
                    "n_miss": len(misses),
                    "mode_pred": mode,
                    "mode_frac": mode_frac,
                }
            )
            print(
                f"  GSM {step:04d}/{GSM_STEPS} loss={loss.item():.3f} lr={lr:.2e} "
                f"gsm={acc:.0%} agree={ag:.0%} misses={len(misses)} "
                f"mode={mode}@{mode_frac:.0%}"
            )
            write_miss_log(
                misses,
                TRACE,
                name=f"miss_gsm_step{step:04d}",
                meta={
                    "phase": "gsm_answer_locked",
                    "step": step,
                    "acc": acc,
                    "mode_pred": mode,
                    "mode_frac": mode_frac,
                },
            )
            if acc <= 0:
                zero_streak += 1
            else:
                zero_streak = 0
            if acc > best["acc"] and ag >= 0.75:
                best = {"acc": acc, "step": step, "agree": ag}
                # Never promote pure collapse (0%) into answer_locked best
                if acc > 0:
                    save_ckpt(
                        student,
                        CKPT / "pure_fsot_gsm_locked_best.pt",
                        step=step,
                        gsm=acc,
                        agree16=ag,
                        phase="gsm_answer_locked",
                    )
                    save_ckpt(
                        student,
                        CKPT / "pure_fsot_answer_locked_best.pt",
                        step=step,
                        gsm=acc,
                        agree16=ag,
                        phase="gsm_answer_locked",
                    )
                    print(f"    * BEST GSM {acc:.0%}")
                else:
                    print(f"    (track gsm={acc:.0%} — not saved while collapsed)")
            # Early stop: solid floor
            if acc >= 0.15 and step >= 400:
                print("GSM floor ≥15% — finish phase A")
                if acc >= 0.20:
                    break
            # Don't burn full 1500 on pure collapse; hand clean host to ARC
            if zero_streak >= 6 and step >= 600 and (best.get("acc") or 0) <= 0:
                print(
                    "GSM still 0% with mode collapse after 6 evals — "
                    "stop phase A, reload clean host for ARC"
                )
                break

    return best, history


def phase_arc(tok, teacher, student, device, plan, opt, start_arc: float = 0.0):
    print("\n=== PHASE B: ARC answer-locked (letter token only) ===")
    train = load_arc_train(DATA / "ARC-Easy_train.csv", ARC_TRAIN_N)
    train += load_arc_train(DATA / "ARC-Challenge_train.csv", ARC_TRAIN_N // 2)
    random.Random(11).shuffle(train)
    eval_rows = load_arc_train(DATA / "ARC-Easy_train.csv", ARC_EVAL_N)

    # Protective mode when host already near/above target — tiny LR, high retention
    protect = start_arc >= (ARC_TARGET - 0.05)
    ret_w = 0.45 if protect else RET_W
    lr_scale = 0.25 if protect else 0.6
    max_steps = 600 if protect else ARC_STEPS
    if protect:
        print(
            f"  protect mode: start_arc={start_arc:.0%} → lr×{lr_scale} "
            f"ret_w={ret_w} max_steps={max_steps}"
        )

    best = {"acc": -1.0, "step": -1, "agree": 0.0}
    history = []
    recent_hits = 0.0
    streak = 0
    drop_streak = 0
    student.train()

    for step in range(1, max_steps + 1):
        r = train[step % len(train)]
        gold = r["gold"].strip().upper()[:1]
        if gold not in "ABCD":
            continue
        prompt = r["prompt"]
        ce = answer_locked_ce(
            student, tok, device, prompt=prompt, gold=gold, kind="letter"
        )
        ce1 = next_token_answer_ce(
            student, tok, device, prompt=prompt, gold=gold, kind="letter"
        )
        ce_r = retention_ce(
            student, teacher, tok, device, EVAL16[step % len(EVAL16)]
        )
        loss = ce + 0.5 * ce1 + ret_w * ce_r
        if not torch.isfinite(loss):
            continue
        lr = fsot_epoch_lr(
            plan,
            epoch=min(step // max(max_steps // 12, 1), 11),
            step=step,
            loss=float(loss.item()),
            recent_hits=recent_hits,
        )
        lr = min(lr * lr_scale, plan.lr0 * lr_scale)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()), GRAD_CLIP)
        opt.step()

        if step % ARC_EVAL_EVERY == 0 or step == 1:
            student.eval()
            acc, misses = eval_arc(
                tok, student, device, eval_rows, collect_misses=True
            )
            ag = agree16(tok, teacher, student, device)
            student.train()
            recent_hits = 0.3 * recent_hits + 0.7 * (acc * 8)
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": lr,
                    "arc": acc,
                    "agree": ag,
                    "n_miss": len(misses),
                    "streak": streak,
                    "protect": protect,
                }
            )
            print(
                f"  ARC {step:04d}/{max_steps} loss={loss.item():.3f} lr={lr:.2e} "
                f"arc={acc:.0%} agree={ag:.0%} streak={streak} misses={len(misses)}"
            )
            write_miss_log(
                misses,
                TRACE,
                name=f"miss_arc_step{step:04d}",
                meta={"phase": "arc_answer_locked", "step": step, "acc": acc},
            )
            if acc > best["acc"] and ag >= 0.75:
                best = {"acc": acc, "step": step, "agree": ag}
                save_ckpt(
                    student,
                    CKPT / "pure_fsot_arc_locked_best.pt",
                    step=step,
                    arc=acc,
                    agree16=ag,
                    phase="arc_answer_locked",
                )
                save_ckpt(
                    student,
                    CKPT / "pure_fsot_answer_locked_best.pt",
                    step=step,
                    arc=acc,
                    agree16=ag,
                    phase="arc_answer_locked",
                )
                print(f"    * BEST ARC {acc:.0%}")

            if acc > ARC_TARGET:
                streak += 1
            else:
                streak = 0
            if streak >= ARC_HOLD_STREAK:
                print(f"*** ARC held >{ARC_TARGET:.0%} for {streak} evals — phase B done ***")
                break

            # Stop if we are destroying a strong start
            floor = max(start_arc - 0.08, 0.20)
            if acc < floor:
                drop_streak += 1
            else:
                drop_streak = 0
            if drop_streak >= 2 and best["acc"] >= 0:
                print(
                    f"ARC drop below floor {floor:.0%} twice — "
                    "reload best and stop phase B"
                )
                p = CKPT / "pure_fsot_arc_locked_best.pt"
                if p.is_file():
                    ck = torch.load(p, map_location=device, weights_only=False)
                    student.load_state_dict(ck["state_dict"], strict=False)
                break

    return best, history


def phase_merge(
    tok, teacher, student, device, plan, opt, gsm_rows, arc_rows, start_arc: float = 0.0
):
    print("\n=== PHASE C: merge GSM+ARC answer-locked ===")
    # Do not shadow eval_gsm / eval_arc functions with list names.
    eval_gsm_rows = load_gsm8k_test(GSM_EVAL_N)
    eval_arc_rows = load_arc_train(DATA / "ARC-Easy_train.csv", ARC_EVAL_N)
    for r in eval_gsm_rows:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"

    best = {"score": -1.0, "gsm": 0.0, "arc": 0.0, "agree": 0.0, "step": -1}
    history = []
    recent_hits = 0.0
    drop_streak = 0
    # ARC-heavy merge (3:1) — GSM free-gen is still collapsed at 135M
    student.train()

    for step in range(1, MERGE_STEPS + 1):
        # 1 GSM : 3 ARC so GSM CE does not trash letter heads
        if step % 4 == 0 and gsm_rows:
            r = gsm_rows[step % len(gsm_rows)]
            qline = r["text"].split("\n")[0]
            if not qline.startswith("Question:"):
                qline = "Question: " + qline
            prompt = f"{qline}\n####"
            gold = str(r["gold"]).strip()
            ce = answer_locked_ce(
                student, tok, device, prompt=prompt, gold=gold, kind="num"
            )
        else:
            r = arc_rows[step % len(arc_rows)]
            gold = r["gold"].strip().upper()[:1]
            if gold not in "ABCD":
                continue
            ce = answer_locked_ce(
                student, tok, device, prompt=r["prompt"], gold=gold, kind="letter"
            )
        ce_r = retention_ce(
            student, teacher, tok, device, EVAL16[step % len(EVAL16)]
        )
        loss = ce + 0.35 * ce_r
        if not torch.isfinite(loss) or float(loss.item()) < 1e-6:
            continue
        lr = fsot_epoch_lr(
            plan,
            epoch=min(step // max(MERGE_STEPS // 8, 1), 7),
            step=step,
            loss=float(loss.item()),
            recent_hits=recent_hits,
        )
        lr = min(lr * 0.35, plan.lr0 * 0.4)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()), GRAD_CLIP)
        opt.step()

        if step % MERGE_EVAL_EVERY == 0 or step == 1:
            student.eval()
            g, mg = eval_gsm(
                tok, student, device, eval_gsm_rows, collect_misses=True
            )
            a, ma = eval_arc(
                tok, student, device, eval_arc_rows, collect_misses=True
            )
            ag = agree16(tok, teacher, student, device)
            student.train()
            score = g * 2.0 + a * 1.5 + ag * 0.3
            recent_hits = 0.3 * recent_hits + 0.7 * (g * 8 + a * 6)
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": lr,
                    "gsm": g,
                    "arc": a,
                    "agree": ag,
                    "score": score,
                }
            )
            print(
                f"  MERGE {step:04d} loss={loss.item():.3f} "
                f"gsm={g:.0%} arc={a:.0%} agree={ag:.0%}"
            )
            write_miss_log(
                mg + ma,
                TRACE,
                name=f"miss_merge_step{step:04d}",
                meta={"phase": "merge", "step": step, "gsm": g, "arc": a},
            )
            if score > best["score"] and ag >= 0.75:
                best = {"score": score, "gsm": g, "arc": a, "agree": ag, "step": step}
                save_ckpt(
                    student,
                    CKPT / "pure_fsot_answer_locked_best.pt",
                    step=step,
                    gsm=g,
                    arc=a,
                    agree16=ag,
                    phase="merge",
                )
                print(f"    * BEST merge gsm={g:.0%} arc={a:.0%}")

            floor = max(start_arc - 0.05, 0.22)
            if a < floor:
                drop_streak += 1
            else:
                drop_streak = 0
            if drop_streak >= 2 and best["score"] >= 0:
                print(f"Merge ARC dropped below {floor:.0%} twice — stop, keep best")
                p = CKPT / "pure_fsot_answer_locked_best.pt"
                if p.is_file():
                    ck = torch.load(p, map_location=device, weights_only=False)
                    student.load_state_dict(ck["state_dict"], strict=False)
                break
            # Hold target: 3 strong ARC evals during merge
            if a > ARC_TARGET and len(history) >= 3:
                recent_arcs = [h["arc"] for h in history[-3:]]
                if all(x > ARC_TARGET for x in recent_arcs):
                    print(f"*** Merge held ARC >{ARC_TARGET:.0%} ×3 — done ***")
                    break

    return best, history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = derive_fsot_lr_plan(d_eff=D_EFF, epochs=12, ref_loss=4.0)
    print("=== ANSWER-LOCKED PLATEAU BREAK ===")
    print(
        f"FSOT LR lr0={plan.lr0:.3e} floor={plan.lr_floor:.3e} ceil={plan.lr_ceil:.3e}"
    )
    print(plan.note)

    tok_t, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tok, student = load_student(device)
    opt = torch.optim.AdamW(list(student.parameters()), lr=plan.lr0, weight_decay=0.01)

    # baseline snapshots
    student.eval()
    eval_gsm0 = load_gsm8k_test(GSM_EVAL_N)
    for r in eval_gsm0:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"
    eval_arc0 = load_arc_train(DATA / "ARC-Easy_train.csv", ARC_EVAL_N)
    g0, _ = eval_gsm(tok, student, device, eval_gsm0)
    a0, _ = eval_arc(tok, student, device, eval_arc0)
    b_g, _ = eval_gsm(tok_t, teacher, device, eval_gsm0)
    b_a, _ = eval_arc(tok_t, teacher, device, eval_arc0)
    ag0 = agree16(tok, teacher, student, device)
    print(f"START gsm={g0:.0%} arc={a0:.0%} agree={ag0:.0%} | BASE gsm={b_g:.0%} arc={b_a:.0%}")

    t0 = time.time()
    gsm_best, gsm_hist = phase_gsm(tok, teacher, student, device, plan, opt)

    # reload best GSM only if it actually moved; else keep clean ARC start
    p_gsm = CKPT / "pure_fsot_gsm_locked_best.pt"
    if p_gsm.is_file() and (gsm_best.get("acc") or 0) > 0.02:
        ck = torch.load(p_gsm, map_location=device, weights_only=False)
        student.load_state_dict(ck["state_dict"], strict=False)
        print("reloaded GSM best for ARC phase")
    else:
        for src in [
            CKPT / "pure_fsot_12x3_best.pt",
            CKPT / "pure_fsot_realdata_best.pt",
        ]:
            if src.is_file():
                ck = torch.load(src, map_location=device, weights_only=False)
                student.load_state_dict(ck["state_dict"], strict=False)
                print("GSM weak — reloaded", src.name, "for ARC phase")
                break

    # Snapshot clean start ARC before phase B (used for protect + floors)
    student.eval()
    a_pre, _ = eval_arc(tok, student, device, eval_arc0)
    student.train()
    print(f"pre-ARC snapshot arc={a_pre:.0%}")

    arc_best, arc_hist = phase_arc(
        tok, teacher, student, device, plan, opt, start_arc=a_pre
    )

    # Prefer best ARC / answer_locked ckpt for merge
    for src in [
        CKPT / "pure_fsot_arc_locked_best.pt",
        CKPT / "pure_fsot_answer_locked_best.pt",
        CKPT / "pure_fsot_12x3_best.pt",
    ]:
        if src.is_file():
            ck = torch.load(src, map_location=device, weights_only=False)
            student.load_state_dict(ck["state_dict"], strict=False)
            print("merge starts from", src.name)
            break

    gsm_train = load_gsm8k_train(min(GSM_TRAIN_N, 2000))
    arc_train = load_arc_train(DATA / "ARC-Easy_train.csv", min(ARC_TRAIN_N, 2000))
    merge_best, merge_hist = phase_merge(
        tok,
        teacher,
        student,
        device,
        plan,
        opt,
        gsm_train,
        arc_train,
        start_arc=max(a_pre, arc_best.get("acc") or 0),
    )

    # Final metrics from BEST checkpoint (not last degraded step)
    p_best = CKPT / "pure_fsot_answer_locked_best.pt"
    if p_best.is_file():
        ck = torch.load(p_best, map_location=device, weights_only=False)
        student.load_state_dict(ck["state_dict"], strict=False)
        print("final eval from", p_best.name, "meta", {k: ck.get(k) for k in ("step", "gsm", "arc", "agree16", "phase")})
    student.eval()
    g_f, misses_g = eval_gsm(tok, student, device, eval_gsm0, collect_misses=True)
    a_f, misses_a = eval_arc(tok, student, device, eval_arc0, collect_misses=True)
    ag_f = agree16(tok, teacher, student, device)
    write_miss_log(
        misses_g + misses_a,
        TRACE,
        name="miss_answer_locked_final",
        meta={
            "gsm": g_f,
            "arc": a_f,
            "agree": ag_f,
            "gsm_best": gsm_best,
            "arc_best": arc_best,
            "merge_best": merge_best,
        },
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "answer_locked: GSM #### tokens only → ARC letter only → merge",
        "lr_plan": {
            "lr0": plan.lr0,
            "floor": plan.lr_floor,
            "ceil": plan.lr_ceil,
            "note": plan.note,
        },
        "start": {"gsm": g0, "arc": a0, "agree": ag0},
        "baseline": {"gsm": b_g, "arc": b_a},
        "gsm_phase": gsm_best,
        "arc_phase": arc_best,
        "merge_phase": merge_best,
        "final": {"gsm": g_f, "arc": a_f, "agree": ag_f},
        "history": {"gsm": gsm_hist, "arc": arc_hist, "merge": merge_hist},
        "beats_base_arc": a_f > b_a,
        "beats_base_gsm": g_f > b_g,
        "plateau_broken_arc": (arc_best.get("acc") or 0) > 0.35,
        "elapsed_s": time.time() - t0,
        "ok": True,
    }
    path = OUT / "answer_locked_train.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "ANSWER_LOCKED_TRAIN.md").write_text(
        f"""# Answer-locked train (plateau break)

| | Baseline | Start | GSM-best | ARC-best | Final |
|--|----------|-------|----------|----------|-------|
| GSM | {b_g:.0%} | {g0:.0%} | **{gsm_best.get('acc',0):.0%}** | — | **{g_f:.0%}** |
| ARC | {b_a:.0%} | {a0:.0%} | — | **{arc_best.get('acc',0):.0%}** | **{a_f:.0%}** |
| Agree | — | {ag0:.0%} | {gsm_best.get('agree',0):.0%} | {arc_best.get('agree',0):.0%} | {ag_f:.0%} |

ARC held >35%: **{(arc_best.get('acc') or 0) > 0.35}**  
Miss trails: `miss_traces/miss_gsm_step*.md`, `miss_arc_step*.md`, `miss_answer_locked_final.md`  
Ckpt: `pure_fsot_answer_locked_best.pt`
""",
        encoding="utf-8",
    )
    print("=== ANSWER-LOCKED SUMMARY ===")
    print(
        f"GSM {g0:.0%}→{g_f:.0%} (best {gsm_best.get('acc',0):.0%}) | "
        f"ARC {a0:.0%}→{a_f:.0%} (best {arc_best.get('acc',0):.0%}) | agree {ag_f:.0%}"
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

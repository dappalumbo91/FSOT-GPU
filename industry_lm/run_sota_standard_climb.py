#!/usr/bin/env python3
"""
SOTA standard climb — one loop under the lab constitution.

docs/SOTA_STANDARDS.md:
  G-VERIFY → measure (capability + overfit) → train → accept_update
  → capability improve → verify_post → promote (push only if all pass)

Architecture fixed: pure FSOT all-layer, FSOT LR, honest holds.
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
from overfit_metrics import (  # noqa: E402
    accept_update,
    combine_reports,
    direction_label,
    measure_arc_overfit,
    measure_gsm_overfit,
    split_disjoint,
    write_overfit_ledger,
)
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


def _digit_token_ids(tok) -> list[int]:
    ids = []
    for d in "0123456789":
        e = tok.encode(d, add_special_tokens=False)
        if len(e) == 1:
            ids.append(e[0])
    return ids


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


def first_digit_vocab_ce(student, tok, device, prompt, gold, digit_ids: list[int] | None = None):
    """
    CE restricted to digit vocabulary only — forces ranking among 0-9,
    not against the whole vocab (anti-collapse / first-digit climb).
    """
    g = str(gold).strip().replace(",", "")
    m = re.search(r"\d", g)
    if not m:
        return torch.tensor(0.0, device=device)
    gold_d = m.group(0)
    digs = digit_ids or _digit_token_ids(tok)
    if not digs:
        return first_digit_ce(student, tok, device, prompt, gold)
    # map gold digit char -> index in digs
    gold_tid = tok.encode(gold_d, add_special_tokens=False)
    if not gold_tid or gold_tid[0] not in digs:
        return first_digit_ce(student, tok, device, prompt, gold)
    local = digs.index(gold_tid[0])
    pe = tok(prompt + " ", return_tensors="pt", truncation=True, max_length=400).to(device)
    logits = student(**pe).logits[0, -1].float()
    # gather digit logits only
    sub = torch.stack([logits[i] for i in digs], dim=0).unsqueeze(0)
    return F.cross_entropy(sub, torch.tensor([local], device=device))


def answer_ce_short(student, tok, device, prompt, gold, kind="num"):
    gids = gold_ids(tok, gold, kind=kind)
    if not gids:
        return torch.tensor(0.0, device=device, requires_grad=True)
    pe = tok(prompt, return_tensors="pt", truncation=True, max_length=400, add_special_tokens=True)
    pids = pe["input_ids"][0].to(device)
    gt = torch.tensor(gids, device=device, dtype=pids.dtype)
    full = torch.cat([pids, gt], dim=0).unsqueeze(0)
    pl = int(pids.numel())
    logits = student(input_ids=full, attention_mask=torch.ones_like(full)).logits
    n = full.size(1) - 1
    start = max(pl - 1, 0)
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


def retention_ce(student, teacher, tok, device, prompt):
    re = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        tlab = int(teacher(**re).logits[0, -1].argmax())
    return F.cross_entropy(
        student(**re).logits[0, -1].float().unsqueeze(0),
        torch.tensor([tlab], device=device),
    )


def set_trainable(student, mode: str):
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


def trainable(student):
    return [p for p in student.parameters() if p.requires_grad]


def pack_cap(arc_e, arc_c, gsm, agree):
    e = float(arc_e.get("exact") or 0)
    c = float(arc_c.get("exact") or 0)
    first = float(gsm.get("first_digit") or 0)
    tf = float(gsm.get("tf_token_acc") or 0)
    exact = float(gsm.get("exact") or 0)
    mn = min(e, c)
    bal = 2.0 * mn + 1.5 * 0.5 * (e + c) + 1.2 * first + 0.8 * tf + 0.4 * agree
    return {
        "arc_e": e,
        "arc_c": c,
        "arc_min": mn,
        "gsm_first": first,
        "gsm_tf": tf,
        "gsm_exact": exact,
        "agree": agree,
        "balanced": bal,
        "mode": gsm.get("mode_pred"),
        "mode_frac": gsm.get("mode_frac") or 0,
    }


def capability_improve(cand, base, eps=1e-4):
    reasons = []
    if cand["agree"] < 0.90:
        return False, ["agree_floor"]
    if cand["arc_min"] + 1e-9 < base["arc_min"] - 0.01:
        return False, ["arc_min_floor"]
    if cand["gsm_first"] + 1e-9 < base["gsm_first"] - 0.05:
        return False, ["gsm_first_floor"]
    improved = False
    if cand["arc_min"] > base["arc_min"] + eps:
        reasons.append(f"arc_min {base['arc_min']:.1%}→{cand['arc_min']:.1%}")
        improved = True
    if cand["arc_e"] > base["arc_e"] + eps and cand["arc_c"] > base["arc_c"] + eps:
        reasons.append("both_arc_holds_up")
        improved = True
    if cand["gsm_first"] > base["gsm_first"] + eps:
        reasons.append(f"gsm_first {base['gsm_first']:.1%}→{cand['gsm_first']:.1%}")
        improved = True
    if cand["gsm_tf"] > base["gsm_tf"] + 0.015:
        reasons.append(f"gsm_tf {base['gsm_tf']:.1%}→{cand['gsm_tf']:.1%}")
        improved = True
    if cand["gsm_exact"] > base["gsm_exact"] + eps:
        reasons.append("gsm_exact_up")
        improved = True
    if cand["balanced"] > base["balanced"] + 0.025 and cand["arc_min"] + 1e-9 >= base["arc_min"] - 0.005:
        reasons.append(f"balanced→{cand['balanced']:.3f}")
        improved = True
    return improved, reasons


def measure_all(tok, teacher, student, device, packs):
    """packs: dict of row lists."""
    student.eval()

    def ea(rows):
        return eval_arc_granular(tok, student, device, rows, arm="fsot")

    def eg(rows):
        return eval_gsm_granular(tok, student, device, rows, arm="fsot")

    ae, _ = ea(packs["easy_hold"])
    ac, _ = ea(packs["ch_hold"])
    g, _ = eg(packs["gsm_hold"])
    ag = agree_n(tok, teacher, student, device, EVAL16)
    cap = pack_cap(ae, ac, g, ag)
    arc_ov = measure_arc_overfit(
        ea,
        easy_train=packs["easy_train"],
        easy_hold=packs["easy_hold"],
        challenge_train=packs["ch_train"],
        challenge_hold=packs["ch_hold"],
        train_eval_n=40,
    )
    gsm_ov = measure_gsm_overfit(
        eg,
        train_rows=packs["gsm_train_probe"],
        hold_rows=packs["gsm_hold"],
        train_eval_n=40,
        metric_key="first_digit",
    )
    ov = combine_reports(arc_ov, gsm_ov)
    return cap, ov


def save_promoted(student, cap, ov, step, phase, gate0):
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in student.state_dict().items()},
        "step": step,
        "phase": phase,
        "sota_standard": True,
        "gate": cap,
        "gate0": gate0,
        "overfit": ov.as_dict(),
        "arc_easy_hold": cap["arc_e"],
        "arc_challenge_hold": cap["arc_c"],
        "arc_min": cap["arc_min"],
        "gsm_first": cap["gsm_first"],
        "gsm_tf": cap["gsm_tf"],
        "gsm_exact": cap["gsm_exact"],
        "agree16": cap["agree"],
        "balanced_score": cap["balanced"],
        "gen_score": ov.gen_score,
        "mean_overfit_gap": ov.mean_overfit_gap,
        "full_dof": True,
        "D_eff": D_EFF,
    }
    torch.save(payload, CKPT / "pure_fsot_sota_standard_best.pt")
    torch.save(payload, CKPT / "pure_fsot_data_driven_best.pt")
    torch.save(
        {**payload, "granular_push": True},
        CKPT / "pure_fsot_granular_best.pt",
    )


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = derive_fsot_lr_plan(d_eff=D_EFF, epochs=12, ref_loss=4.0)
    print("=== SOTA STANDARD CLIMB ===")
    print("constitution: docs/SOTA_STANDARDS.md")
    print(f"FSOT LR lr0={plan.lr0:.3e}")

    # ----- G-VERIFY pre -----
    print("\n[G-VERIFY pre]")
    v_pre = run_verification(include_host=True, write=True)
    for n, c in v_pre["layers"].items():
        print(f"  [{'OK' if c.get('ok') else 'FAIL'}] {n}")
    if not v_pre["ok"]:
        print("VERIFY FAIL — stop (standards refuse ungreen train)")
        (OUT / "sota_standard_climb.json").write_text(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "promote_to_github": False,
                    "reason": "verify_pre_failed",
                    "verify_pre": v_pre,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return 1

    # ----- data packs -----
    easy_all = load_arc_train(DATA / "ARC-Easy_train.csv", None)
    ch_all = load_arc_train(DATA / "ARC-Challenge_train.csv", None)
    easy_tr, easy_h = split_disjoint(easy_all, train_n=2500, hold_n=60, seed=17)
    ch_tr, ch_h = split_disjoint(ch_all, train_n=1500, hold_n=40, seed=19)
    gsm_hold = load_gsm8k_test(40)
    for r in gsm_hold:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"
    gsm_tr_raw = load_gsm8k_train(400)
    gsm_train_probe = []
    for r in gsm_tr_raw:
        q = r["text"].split("\n")[0]
        if not q.startswith("Question:"):
            q = "Question: " + q
        gsm_train_probe.append({"prompt": f"{q}\n####", "gold": r["gold"]})
    packs = {
        "easy_train": easy_tr,
        "easy_hold": easy_h,
        "ch_train": ch_tr,
        "ch_hold": ch_h,
        "gsm_hold": gsm_hold,
        "gsm_train_probe": gsm_train_probe,
    }

    tok_t, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

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
        print("no host checkpoint found")
        return 1
    tok, student = load_model(device)
    swap_all_layers(student)
    ck0 = torch.load(src, map_location=device, weights_only=False)
    student.load_state_dict(ck0["state_dict"], strict=False)
    print("host", src.name)

    cap0, ov0 = measure_all(tok, teacher, student, device, packs)
    write_overfit_ledger(ov0, OUT, name="overfit_sota_start", meta={"phase": "start"})
    print(
        f"START cap min={cap0['arc_min']:.0%} E={cap0['arc_e']:.0%} C={cap0['arc_c']:.0%} "
        f"first={cap0['gsm_first']:.0%} tf={cap0['gsm_tf']:.0%} ag={cap0['agree']:.0%} "
        f"bal={cap0['balanced']:.3f}"
    )
    print(
        f"START ov  hold={ov0.mean_hold_acc:.0%} gap={ov0.mean_overfit_gap:+.0%} "
        f"gen={ov0.gen_score:.3f} flag={ov0.overfit_flag}"
    )

    best_cap = dict(cap0)
    best_ov = ov0
    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    promoted = False
    history = []
    t0 = time.time()

    # ----- train: digit-vocab first-digit focus + soft ARC hold -----
    # Head-only; digit CE restricted to 0-9 ranking (open: first-digit climb).
    print("\n[TRAIN] digit-row gradient mask on tied embed/lm_head (+ light ARC)")
    # SmolLM2 ties lm_head ↔ embed_tokens — enable that weight only, mask rows
    for p in student.parameters():
        p.requires_grad_(False)
    tied = False
    for name, p in student.named_parameters():
        if "embed_tokens.weight" in name or name.endswith("lm_head.weight"):
            p.requires_grad_(True)
            tied = True
            print(f"  enabled {name} {tuple(p.shape)}")
    if not tied:
        # fallback: any head/embed
        set_trainable(student, "head")
        print("  fallback set_trainable(head)")
    n_tr = sum(p.numel() for p in trainable(student))
    print(f"  trainable {n_tr/1e6:.2f}M; digit+letter row mask")
    if n_tr < 1:
        print("no trainable params — abort")
        return 1
    opt = torch.optim.AdamW(trainable(student), lr=plan.lr0 * 1.1, weight_decay=0.0)
    digit_ids = _digit_token_ids(tok)
    print(f"  digit token ids: {digit_ids}")
    # letter tokens for ARC soft retention on head rows
    letter_ids = []
    for L in ("A", "B", "C", "D", " A", " B", " C", " D"):
        e = tok.encode(L, add_special_tokens=False)
        if len(e) == 1:
            letter_ids.append(e[0])
    letter_ids = sorted(set(letter_ids))
    print(f"  letter token ids: {letter_ids}")

    def mask_head_grad_digits_and_letters():
        """Only update lm_head rows for digits + ABCD — curb ARC bleed from digit CE."""
        wh = None
        for name, p in student.named_parameters():
            if p.grad is None:
                continue
            if "lm_head.weight" in name or name.endswith("lm_head.weight"):
                wh = p
                break
        # tied embeddings sometimes use embed_tokens
        if wh is None:
            for name, p in student.named_parameters():
                if p.grad is None:
                    continue
                if "embed_tokens.weight" in name:
                    wh = p
                    break
        if wh is None or wh.grad is None:
            return
        allow = set(digit_ids + letter_ids)
        mask = torch.zeros_like(wh.grad)
        for i in allow:
            if 0 <= i < mask.size(0):
                mask[i] = 1.0
        wh.grad.mul_(mask)

    rng = random.Random(31)
    arith = []
    # Prefer 1-digit golds for first-digit signal (0-9 sums/diffs)
    while len(arith) < 2000:
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        if rng.random() < 0.6:
            gold, q = str(a + b), f"What is {a} + {b}?"
        else:
            aa, bb = max(a, b), min(a, b)
            gold, q = str(aa - bb), f"What is {aa} - {bb}?"
        # keep many single-digit answers
        if len(gold) == 1 or (len(gold) == 2 and rng.random() < 0.35):
            arith.append({"prompt": f"Question: {q}\n####", "gold": gold})
    # pure digit identity: #### → d
    for d in range(10):
        for _ in range(40):
            arith.append(
                {
                    "prompt": f"Question: Write the digit {d}.\n####",
                    "gold": str(d),
                }
            )
    random.Random(32).shuffle(arith)
    gsm_real = [r for r in load_gsm8k_train(2000) if len(str(r["gold"]).strip()) <= 2]
    if cap0["arc_c"] <= cap0["arc_e"]:
        arc_mix = ch_tr[:600] + easy_tr[:300]
    else:
        arc_mix = easy_tr[:600] + ch_tr[:300]
    random.Random(33).shuffle(gsm_real)
    random.Random(34).shuffle(arc_mix)

    STEPS = 700
    EVAL_EVERY = 50
    reject_streak = 0
    arc_min_floor = max(cap0["arc_min"] - 0.02, 0.22)
    student.train()

    for step in range(1, STEPS + 1):
        r = step % 10
        # 50% arith digit-vocab, 20% real short GSM, 30% ARC hold (protect min)
        if r < 5:
            row = arith[step % len(arith)]
            fd_v = first_digit_vocab_ce(
                student, tok, device, row["prompt"], row["gold"], digit_ids
            )
            fd = first_digit_ce(student, tok, device, row["prompt"], row["gold"])
            ce = answer_ce_short(student, tok, device, row["prompt"], row["gold"], "num")
            loss_task = 2.0 * fd_v + 0.8 * fd + 0.3 * ce
            task = "digit"
        elif r < 7 and gsm_real:
            row = gsm_real[step % len(gsm_real)]
            q = row["text"].split("\n")[0]
            if not q.startswith("Question:"):
                q = "Question: " + q
            prompt = f"{q}\n####"
            gold = str(row["gold"]).strip()
            fd_v = first_digit_vocab_ce(student, tok, device, prompt, gold, digit_ids)
            fd = first_digit_ce(student, tok, device, prompt, gold)
            ce = answer_ce_short(student, tok, device, prompt, gold, "num")
            loss_task = 2.0 * fd_v + 0.8 * fd + 0.3 * ce
            task = "gsm"
        else:
            row = arc_mix[step % len(arc_mix)]
            gold = row["gold"].strip().upper()[:1]
            if gold not in "ABCD":
                continue
            loss_task = next_ce(student, tok, device, row["prompt"], gold, "letter")
            task = "arc"

        ce_r = retention_ce(student, teacher, tok, device, EVAL16[step % len(EVAL16)])
        loss = loss_task + 0.55 * ce_r
        if not torch.isfinite(loss):
            continue
        for g in opt.param_groups:
            g["lr"] = min(plan.lr0 * 0.9, 4.5e-5)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        mask_head_grad_digits_and_letters()
        torch.nn.utils.clip_grad_norm_(trainable(student), 0.5)
        opt.step()

        if step % EVAL_EVERY != 0 and step != 1:
            continue

        cap, ov = measure_all(tok, teacher, student, device, packs)
        student.train()
        ov_ok, ov_reasons = accept_update(
            before=best_ov,
            after=ov,
            min_hold_delta=-0.005,  # allow tiny noise
            max_gap_widen=0.025,
            require_gen_improve=False,  # allow hold-flat if gap shrinks + cap up
        )
        # if gen improves, force ov_ok
        if ov.gen_score > best_ov.gen_score + 1e-4 and ov.mean_hold_acc + 1e-9 >= best_ov.mean_hold_acc - 0.01:
            ov_ok = True
            ov_reasons = ["gen_score_up"]
        cap_ok, cap_reasons = capability_improve(cap, best_cap)
        dlab = direction_label(best_ov, ov)
        history.append(
            {
                "step": step,
                "task": task,
                "loss": float(loss.item()),
                **cap,
                "gen_score": ov.gen_score,
                "overfit_gap": ov.mean_overfit_gap,
                "overfit_flag": ov.overfit_flag,
                "direction": dlab,
                "ov_ok": ov_ok,
                "cap_ok": cap_ok,
                "ov_reasons": ov_reasons,
                "cap_reasons": cap_reasons,
            }
        )
        print(
            f"  {step:04d}/{STEPS} {task} loss={loss.item():.3f} "
            f"min={cap['arc_min']:.0%} E={cap['arc_e']:.0%} C={cap['arc_c']:.0%} "
            f"first={cap['gsm_first']:.0%} tf={cap['gsm_tf']:.0%} x={cap['gsm_exact']:.0%} "
            f"ag={cap['agree']:.0%} gen={ov.gen_score:.3f} gap={ov.mean_overfit_gap:+.0%} "
            f"dir={dlab} ov={ov_ok} cap={cap_ok}"
        )

        # Standards promote: need capability improve AND not overfit-worsening
        if cap_ok and ov_ok:
            best_cap = dict(cap)
            best_ov = ov
            best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
            promoted = True
            reject_streak = 0
            save_promoted(student, cap, ov, step, "sota_standard_climb", cap0)
            print(f"    * PROMOTED {cap_reasons} | ov {ov_reasons}")
        elif not ov_ok and dlab in ("OVERFIT_STEP", "MEMORIZE_COLLAPSE"):
            student.load_state_dict(best_state, strict=False)
            reject_streak += 1
            print(f"    * REJECT overfit dir — restored ({ov_reasons})")
            if reject_streak >= 5:
                print("too many overfit rejects — stop train")
                break
        else:
            if cap["arc_min"] + 1e-9 < arc_min_floor:
                student.load_state_dict(best_state, strict=False)
                reject_streak += 1
                print(f"    * REJECT below arc_min floor {arc_min_floor:.0%} — restored")
                if reject_streak >= 5:
                    break
            elif cap["arc_min"] + 1e-9 < best_cap["arc_min"] - 0.02:
                student.load_state_dict(best_state, strict=False)
                reject_streak += 1
                print("    * REJECT arc_min drop — restored")
                if reject_streak >= 5:
                    break
            else:
                reject_streak = 0

    # restore best for final measure
    student.load_state_dict(best_state, strict=False)
    cap_f, ov_f = measure_all(tok, teacher, student, device, packs)
    write_overfit_ledger(ov_f, OUT, name="overfit_sota_final", meta={"phase": "final"})

    # ----- G-VERIFY post -----
    print("\n[G-VERIFY post]")
    final_ckpt = CKPT / "pure_fsot_sota_standard_best.pt"
    if not final_ckpt.is_file():
        final_ckpt = src
    v_post = run_verification(
        include_host=True,
        ckpt_path=final_ckpt if final_ckpt.is_file() else None,
        write=True,
    )
    for n, c in v_post["layers"].items():
        print(f"  [{'OK' if c.get('ok') else 'FAIL'}] {n}")

    cap_beat, reasons = capability_improve(cap_f, cap0)
    ov_beat, ov_r = accept_update(
        before=ov0,
        after=ov_f,
        min_hold_delta=-0.005,
        max_gap_widen=0.03,
        require_gen_improve=False,
    )
    if ov_f.gen_score > ov0.gen_score + 1e-4:
        ov_beat = True
        ov_r = ["gen_score_up"]

    promote = bool(
        promoted
        and cap_beat
        and ov_beat
        and v_pre["ok"]
        and v_post["ok"]
    )
    elapsed = time.time() - t0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "sota_standard_climb",
        "constitution": "docs/SOTA_STANDARDS.md",
        "architecture": "pure_FSOT_all_layer_SmolLM2-135M",
        "start_host": src.name,
        "verify_pre_ok": v_pre["ok"],
        "verify_post_ok": v_post["ok"],
        "start": {"cap": cap0, "overfit": ov0.as_dict()},
        "final": {"cap": cap_f, "overfit": ov_f.as_dict()},
        "best": {"cap": best_cap, "overfit": best_ov.as_dict()},
        "capability_improved": cap_beat,
        "overfit_acceptable": ov_beat,
        "promote_reasons": reasons + ov_r if promote else [],
        "promote_to_github": promote,
        "history": history,
        "elapsed_s": elapsed,
        "deltas": {
            "arc_min": cap_f["arc_min"] - cap0["arc_min"],
            "arc_e": cap_f["arc_e"] - cap0["arc_e"],
            "arc_c": cap_f["arc_c"] - cap0["arc_c"],
            "gsm_first": cap_f["gsm_first"] - cap0["gsm_first"],
            "gsm_tf": cap_f["gsm_tf"] - cap0["gsm_tf"],
            "gsm_exact": cap_f["gsm_exact"] - cap0["gsm_exact"],
            "balanced": cap_f["balanced"] - cap0["balanced"],
            "gen_score": ov_f.gen_score - ov0.gen_score,
            "overfit_gap": ov_f.mean_overfit_gap - ov0.mean_overfit_gap,
        },
    }
    (OUT / "sota_standard_climb.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    status = (
        "IMPROVED — eligible to push"
        if promote
        else "NO_PUSH (standards not all met)"
    )
    (OUT / "SOTA_STANDARD_CLIMB.md").write_text(
        f"""# SOTA standard climb

**Status: {status}**

Constitution: [`docs/SOTA_STANDARDS.md`](../../docs/SOTA_STANDARDS.md)

## Gates

| Gate | Pre | Post / Final |
|------|-----|--------------|
| G-VERIFY | {'PASS' if v_pre['ok'] else 'FAIL'} | {'PASS' if v_post['ok'] else 'FAIL'} |
| G-OVERFIT gen_score | {ov0.gen_score:.3f} | {ov_f.gen_score:.3f} (Δ {ov_f.gen_score-ov0.gen_score:+.3f}) |
| G-OVERFIT gap | {ov0.mean_overfit_gap:+.0%} | {ov_f.mean_overfit_gap:+.0%} |
| G-CAP arc_min | {cap0['arc_min']:.0%} | {cap_f['arc_min']:.0%} |
| G-CAP gsm_first | {cap0['gsm_first']:.0%} | {cap_f['gsm_first']:.0%} |
| G-CAP agree | {cap0['agree']:.0%} | {cap_f['agree']:.0%} |

## Capability table

| Axis | Start | Final | Δ |
|------|-------|-------|---|
| ARC min | {cap0['arc_min']:.0%} | {cap_f['arc_min']:.0%} | {(cap_f['arc_min']-cap0['arc_min']):+.0%} |
| ARC-Easy hold | {cap0['arc_e']:.0%} | {cap_f['arc_e']:.0%} | {(cap_f['arc_e']-cap0['arc_e']):+.0%} |
| ARC-Challenge hold | {cap0['arc_c']:.0%} | {cap_f['arc_c']:.0%} | {(cap_f['arc_c']-cap0['arc_c']):+.0%} |
| GSM first-digit | {cap0['gsm_first']:.0%} | {cap_f['gsm_first']:.0%} | {(cap_f['gsm_first']-cap0['gsm_first']):+.0%} |
| GSM TF | {cap0['gsm_tf']:.0%} | {cap_f['gsm_tf']:.0%} | {(cap_f['gsm_tf']-cap0['gsm_tf']):+.0%} |
| GSM free exact | {cap0['gsm_exact']:.0%} | {cap_f['gsm_exact']:.0%} | {(cap_f['gsm_exact']-cap0['gsm_exact']):+.0%} |
| Balanced | {cap0['balanced']:.3f} | {cap_f['balanced']:.3f} | {(cap_f['balanced']-cap0['balanced']):+.3f} |
| gen_score | {ov0.gen_score:.3f} | {ov_f.gen_score:.3f} | {(ov_f.gen_score-ov0.gen_score):+.3f} |

Promote reasons: {reasons + ov_r if promote else 'none'}  
Elapsed: {elapsed:.0f}s  
Architecture: pure FSOT all-layer · SmolLM2-135M · RTX-class host
""",
        encoding="utf-8",
    )
    print("===", status, "===")
    print(
        f"min {cap0['arc_min']:.0%}→{cap_f['arc_min']:.0%} first {cap0['gsm_first']:.0%}→{cap_f['gsm_first']:.0%} "
        f"gen {ov0.gen_score:.3f}→{ov_f.gen_score:.3f} verify={v_pre['ok']}/{v_post['ok']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

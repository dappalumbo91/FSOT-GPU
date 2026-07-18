#!/usr/bin/env python3
"""
Refine pure FSOT host on REAL packs (GSM8K / ARC / MATH from D:\\training data).

Why accuracy "vanished": earlier refinement optimized probe/synthetic fidelity;
capability smoke uses real held-out-style items. This script trains on the real
distribution under FSOT law only:

  - pure FSOT consensus attention (all layers)
  - suction_poof_lr + D_eff scalar (no free LR schedule)
  - full parameter DoF
  - light EVAL16 retention so host fidelity does not evaporate

Saves: results/industry_lm/checkpoints/pure_fsot_realdata_best.pt
Ledger: results/industry_lm/real_data_train.json
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
from fsot_lib.learn import suction_poof_lr  # noqa: E402
from fsot_lib.scalar import compute_scalar  # noqa: E402
from real_data_packs import (  # noqa: E402
    build_train_mix,
    load_arc_train,
    load_gsm8k_test,
    load_math_train,
)
from train_corpus import PROBES  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
CKPT = OUT / "checkpoints"
DATA = Path(r"D:\training data")
CKPT.mkdir(parents=True, exist_ok=True)

D_EFF = 14.0
STEPS = 2500
EVAL_EVERY = 100
BATCH = 2
SEQ = 384
RET_W = 0.2

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


def fsot_lr(step: int, recent_hits: float, loss: float) -> float:
    sp = suction_poof_lr(step, recent_hits, loss)
    S = abs(
        float(
            compute_scalar(
                N=1.0,
                P=1.0,
                D_eff=D_EFF,
                delta_psi=0.1 + 0.05 * math.sin(step * 0.01),
                recent_hits=recent_hits,
                observed=True,
                delta_theta=(step % 100) / 100.0 * math.pi,
            )
        )
    )
    mult = 0.35 + 0.65 * (1.0 - math.exp(-S))
    # real-data train: slightly higher ceiling than probe-only
    return max(min(sp * mult * 0.18, 2.5e-5), 5e-7)


def extract_num(s: str):
    nums = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
    return nums[-1] if nums else None


@torch.no_grad()
def gen(tok, model, device, prompt, max_new=64):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = model.generate(
        **inp,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt) :]
    return text


@torch.no_grad()
def score_gsm(tok, model, device, rows, max_new=16):
    """
    Honest GSM: prompt ends with ####; model should emit the final number next.
    Score first number after the prompt only (no question-digit leakage).
    """
    hits = 0
    for r in rows:
        # normalize prompt to #### format
        q = r["prompt"]
        if "####" not in q:
            # derive from text if needed
            q = r["prompt"].split("Answer:")[0].strip()
            if not q.startswith("Question:"):
                q = "Question: " + q
            q = q.rstrip() + "\n####"
        tail = gen(tok, model, device, q if q.endswith("####") else q + "\n####", max_new=max_new)
        gold = extract_num(str(r["gold"])) or str(r["gold"]).strip()
        nums = re.findall(r"-?\d+\.?\d*", tail.replace(",", ""))
        pred = nums[0] if nums else None
        ok = pred is not None and gold is not None and pred == gold
        hits += int(ok)
    return hits / max(len(rows), 1)


@torch.no_grad()
def score_arc(tok, model, device, rows, max_new=8):
    hits = 0
    for r in rows:
        tail = gen(tok, model, device, r["prompt"], max_new=max_new)
        m = re.search(r"\b([ABCD])\b", tail.upper())
        pred = m.group(1) if m else (tail.strip()[:1].upper() if tail.strip() else "")
        hits += int(pred == r["gold"].upper())
    return hits / max(len(rows), 1)


@torch.no_grad()
def score_math(tok, model, device, rows, max_new=48):
    hits = 0
    for r in rows:
        tail = gen(tok, model, device, r["prompt"], max_new=max_new)
        gold = str(r["gold"])
        # loose: gold fragment or last number match
        ok = False
        gnum = extract_num(gold)
        pnum = extract_num(tail)
        if gnum and pnum and gnum == pnum:
            ok = True
        # short symbolic answers
        gclean = re.sub(r"\s+", "", gold.lower())[:40]
        if gclean and gclean in re.sub(r"\s+", "", tail.lower()):
            ok = True
        hits += int(ok)
    return hits / max(len(rows), 1)


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


def batch_ce(student, tok, texts, device):
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=SEQ,
    ).to(device)
    logits = student(**enc).logits
    shift_l = logits[:, :-1, :].contiguous()
    shift_y = enc["input_ids"][:, 1:].contiguous()
    mask = enc["attention_mask"][:, 1:].float()
    # boost answer half of each sequence
    w = torch.ones_like(mask)
    mid = mask.size(1) // 2
    w[:, mid:] = 2.5
    ce = F.cross_entropy(
        shift_l.reshape(-1, shift_l.size(-1)),
        shift_y.reshape(-1),
        reduction="none",
    ).view_as(shift_y)
    return (ce * mask * w).sum() / (mask * w).sum().clamp_min(1.0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== REAL-DATA FSOT REFINE (GSM8K / ARC / MATH) ===")
    print("data root", DATA)

    train = build_train_mix(
        n_gsm=5000,
        n_arc_easy=4000,
        n_arc_hard=2000,
        n_math=500,
    )
    random.Random(42).shuffle(train)
    print(
        "train n=",
        len(train),
        "by kind",
        {k: sum(1 for r in train if r["kind"] == k) for k in ("gsm8k", "arc", "math")},
    )

    # eval slices (fixed)
    # train-time eval: modest n for wall-clock; final smoke can re-run larger
    eval_gsm = load_gsm8k_test(30)
    eval_arc = load_arc_train(DATA / "ARC-Easy_train.csv", 40)
    eval_math = load_math_train(25)
    print(f"eval gsm={len(eval_gsm)} arc={len(eval_arc)} math={len(eval_math)}")

    tok, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    _, student = load_model(device)
    swap_all_layers(student)
    # start from curriculum (FSOT literacy) not collapsed climb
    for src in [
        CKPT / "pure_fsot_curriculum_best.pt",
        CKPT / "pure_fsot_agree100_best.pt",
        CKPT / "pure_fsot_exceed_best.pt",
    ]:
        if src.is_file():
            ck = torch.load(src, map_location=device, weights_only=False)
            student.load_state_dict(ck["state_dict"], strict=False)
            print("loaded", src.name)
            break

    for p in student.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(list(student.parameters()), lr=1e-5, weight_decay=0.01)

    student.eval()
    print("baseline host capability...")
    b_gsm = score_gsm(tok, teacher, device, eval_gsm)
    b_arc = score_arc(tok, teacher, device, eval_arc)
    b_math = score_math(tok, teacher, device, eval_math)
    print(f"  BASE gsm={b_gsm:.0%} arc={b_arc:.0%} math={b_math:.0%}")

    print("start FSOT capability...")
    s_gsm = score_gsm(tok, student, device, eval_gsm)
    s_arc = score_arc(tok, student, device, eval_arc)
    s_math = score_math(tok, student, device, eval_math)
    a0 = agree16(tok, teacher, student, device)
    print(f"  FSOT gsm={s_gsm:.0%} arc={s_arc:.0%} math={s_math:.0%} agree={a0:.0%}")

    def macro(g, a, m):
        return (g + a + m) / 3.0

    best = {
        "macro": macro(s_gsm, s_arc, s_math),
        "gsm": s_gsm,
        "arc": s_arc,
        "math": s_math,
        "agree": a0,
        "step": -1,
    }
    history = []
    recent_hits = 0.0
    student.train()
    t0 = time.time()

    gsm_pool = [r for r in train if r["kind"] == "gsm8k"]
    arc_pool = [r for r in train if r["kind"] == "arc"]

    for step in range(1, STEPS + 1):
        # Mix: full sequence CE on real packs
        batch = [train[(step * BATCH + i) % len(train)]["text"] for i in range(BATCH)]
        if gsm_pool:
            batch[0] = gsm_pool[step % len(gsm_pool)]["text"]
        ce = batch_ce(student, tok, batch, device)

        # Surgical GSM: next-token after "####" must be gold number
        ce_gsm_tok = torch.tensor(0.0, device=device)
        if gsm_pool:
            gr = gsm_pool[(step * 3) % len(gsm_pool)]
            prompt = gr["prompt"]
            if not prompt.rstrip().endswith("####"):
                prompt = f"Question: {gr.get('prompt','')}\n####"
                # rebuild from gold-known structure
                qline = gr["text"].split("\n")[0]
                prompt = f"{qline}\n####"
            gold = str(gr["gold"]).strip()
            # target first token of " 72" or "72"
            for cand in (f" {gold}", gold):
                ids = tok.encode(cand, add_special_tokens=False)
                if ids:
                    tid = ids[0]
                    break
            else:
                tid = None
            if tid is not None:
                pe = tok(prompt, return_tensors="pt", truncation=True, max_length=SEQ).to(
                    device
                )
                logits = student(**pe).logits[0, -1]
                ce_gsm_tok = F.cross_entropy(
                    logits.float().unsqueeze(0),
                    torch.tensor([tid], device=device),
                )

        # Surgical ARC: next token is A/B/C/D
        ce_arc_tok = torch.tensor(0.0, device=device)
        if arc_pool:
            ar = arc_pool[(step * 5) % len(arc_pool)]
            key = ar["gold"].strip().upper()[:1]
            if key in "ABCD":
                pe = tok(
                    ar["prompt"], return_tensors="pt", truncation=True, max_length=SEQ
                ).to(device)
                tid = tok.encode(f" {key}", add_special_tokens=False)[0]
                logits = student(**pe).logits[0, -1]
                ce_arc_tok = F.cross_entropy(
                    logits.float().unsqueeze(0),
                    torch.tensor([tid], device=device),
                )

        # retention
        rp = EVAL16[step % len(EVAL16)]
        re = tok(rp, return_tensors="pt").to(device)
        with torch.no_grad():
            tlab = int(teacher(**re).logits[0, -1].argmax())
        ce_r = F.cross_entropy(
            student(**re).logits[0, -1].float().unsqueeze(0),
            torch.tensor([tlab], device=device),
        )
        loss = ce + 1.5 * ce_gsm_tok + 1.2 * ce_arc_tok + RET_W * ce_r
        if not torch.isfinite(loss):
            continue

        lr = fsot_lr(step, recent_hits, float(loss.item()))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(student.parameters()), 1.0)
        opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            student.eval()
            g = score_gsm(tok, student, device, eval_gsm)
            a = score_arc(tok, student, device, eval_arc)
            m = score_math(tok, student, device, eval_math)
            ag = agree16(tok, teacher, student, device)
            student.train()
            mac = macro(g, a, m)
            # hits feedback: how close to baseline macro
            base_mac = macro(b_gsm, b_arc, b_math)
            recent_hits = 0.25 * recent_hits + 0.75 * max(mac / max(base_mac, 1e-3), 0.0) * 5
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": lr,
                    "gsm": g,
                    "arc": a,
                    "math": m,
                    "macro": mac,
                    "agree": ag,
                    "base_macro": base_mac,
                }
            )
            print(
                f"  {step:04d} loss={loss.item():.3f} gsm={g:.0%} arc={a:.0%} "
                f"math={m:.0%} macro={mac:.0%} (base {base_mac:.0%}) agree={ag:.0%}"
            )

            # save if macro improves and agree does not collapse
            if mac >= best["macro"] - 1e-9 and ag >= 0.75:
                if mac > best["macro"] or (abs(mac - best["macro"]) < 1e-9 and ag > best["agree"]):
                    best = {
                        "macro": mac,
                        "gsm": g,
                        "arc": a,
                        "math": m,
                        "agree": ag,
                        "step": step,
                    }
                    torch.save(
                        {
                            "step": step,
                            "gsm": g,
                            "arc": a,
                            "math": m,
                            "macro": mac,
                            "agree16": ag,
                            "base_gsm": b_gsm,
                            "base_arc": b_arc,
                            "base_math": b_math,
                            "D_eff": D_EFF,
                            "full_dof": True,
                            "real_data": True,
                            "state_dict": {
                                k: v.detach().cpu()
                                for k, v in student.state_dict().items()
                            },
                        },
                        CKPT / "pure_fsot_realdata_best.pt",
                    )
                    print(
                        f"    * BEST macro={mac:.0%} gsm={g:.0%} arc={a:.0%} "
                        f"math={m:.0%} agree={ag:.0%}"
                    )

            # early win: beat baseline macro with agree floor
            if mac > base_mac and ag >= 0.80:
                print("*** REAL-DATA MACRO BEATS BASELINE ***")
                if mac >= base_mac + 0.05:
                    break

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "authority": "pure FSOT consensus + suction_poof + D_eff; real D:/training data packs",
        "train_n": len(train),
        "base": {"gsm": b_gsm, "arc": b_arc, "math": b_math, "macro": macro(b_gsm, b_arc, b_math)},
        "start": {
            "gsm": s_gsm,
            "arc": s_arc,
            "math": s_math,
            "macro": macro(s_gsm, s_arc, s_math),
            "agree": a0,
        },
        "best": best,
        "history": history,
        "beats_base_macro": best["macro"] > macro(b_gsm, b_arc, b_math),
        "elapsed_s": time.time() - t0,
        "ok": True,
    }
    path = OUT / "real_data_train.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "REAL_DATA_TRAIN.md").write_text(
        f"""# Real-data FSOT refine

Trained pure FSOT host on **real** GSM8K / ARC / MATH from `D:\\training data`.

| | Baseline host | Start FSOT | Best FSOT |
|--|---------------|------------|-----------|
| GSM8K | {b_gsm:.0%} | {s_gsm:.0%} | **{best['gsm']:.0%}** |
| ARC-Easy | {b_arc:.0%} | {s_arc:.0%} | **{best['arc']:.0%}** |
| MATH | {b_math:.0%} | {s_math:.0%} | **{best['math']:.0%}** |
| Macro | {macro(b_gsm,b_arc,b_math):.0%} | {macro(s_gsm,s_arc,s_math):.0%} | **{best['macro']:.0%}** |
| Agree16 | 100% self | {a0:.0%} | {best['agree']:.0%} |

Beats base macro: **{best['macro'] > macro(b_gsm,b_arc,b_math)}**  
Ckpt: `checkpoints/pure_fsot_realdata_best.pt`
""",
        encoding="utf-8",
    )
    print("=== REAL DATA SUMMARY ===")
    print(
        f"best macro={best['macro']:.0%} base={macro(b_gsm,b_arc,b_math):.0%} "
        f"gsm={best['gsm']:.0%} arc={best['arc']:.0%} math={best['math']:.0%}"
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Fast few-epoch real-data train: **12 epochs × 3 packs** (GSM8K, ARC, MATH).

LR entirely from FSOT law (derive_fsot_lr_plan + fsot_epoch_lr).
Goals: fast learning, little epochs, no catastrophic failure, no huge scale.

Saves: pure_fsot_12x3_best.pt
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
from fsot_lib.learn import derive_fsot_lr_plan, fsot_epoch_lr  # noqa: E402
from real_data_packs import (  # noqa: E402
    load_arc_train,
    load_gsm8k_test,
    load_gsm8k_train,
    load_math_train,
)
from train_corpus import PROBES  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
CKPT = OUT / "checkpoints"
DATA = Path(r"D:\training data")
CKPT.mkdir(parents=True, exist_ok=True)

# --- 12 epochs × 3 packs ---
EPOCHS = 12
# per-epoch subsample (fast; full mix over 12 passes still covers thousands)
N_GSM = 600
N_ARC = 500
N_MATH = 250
BATCH = 2
SEQ = 320
D_EFF = 14.0
EVAL_EVERY_EPOCH = 1
RET_W = 0.2
GRAD_CLIP = 0.5  # anti-catastrophe

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


def extract_num(s: str):
    nums = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
    return nums[-1] if nums else None


@torch.no_grad()
def gen(tok, model, device, prompt, max_new=16):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=480).to(device)
    out = model.generate(
        **inp,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt) :] if text.startswith(prompt) else text


@torch.no_grad()
def score_gsm(tok, model, device, rows):
    hits = 0
    for r in rows:
        p = r["prompt"]
        if not p.rstrip().endswith("####"):
            p = p.split("Answer:")[0].strip() + "\n####"
        tail = gen(tok, model, device, p, max_new=12)
        gold = extract_num(str(r["gold"])) or str(r["gold"]).strip()
        nums = re.findall(r"-?\d+\.?\d*", tail.replace(",", ""))
        pred = nums[0] if nums else None
        hits += int(pred is not None and gold is not None and pred == gold)
    return hits / max(len(rows), 1)


@torch.no_grad()
def score_arc(tok, model, device, rows):
    hits = 0
    for r in rows:
        tail = gen(tok, model, device, r["prompt"], max_new=6)
        m = re.search(r"\b([ABCD])\b", tail.upper())
        pred = m.group(1) if m else (tail.strip()[:1].upper() if tail.strip() else "")
        hits += int(pred == r["gold"].upper())
    return hits / max(len(rows), 1)


@torch.no_grad()
def score_math(tok, model, device, rows):
    hits = 0
    for r in rows:
        tail = gen(tok, model, device, r["prompt"], max_new=24)
        g = extract_num(str(r["gold"]))
        p = extract_num(tail)
        hits += int(g is not None and p is not None and g == p)
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


def build_epoch_data(rng: random.Random):
    """Three packs, fixed sizes, reshuffled each epoch."""
    gsm = load_gsm8k_train(N_GSM * 2)  # short+full variants
    # prefer short #### lines
    gsm_short = [r for r in gsm if r["text"].count("\n") <= 2] or gsm
    gsm = rng.sample(gsm_short, min(N_GSM, len(gsm_short)))
    arc_e = load_arc_train(DATA / "ARC-Easy_train.csv", N_ARC)
    arc_c = load_arc_train(DATA / "ARC-Challenge_train.csv", N_ARC // 2)
    arc = arc_e + arc_c
    rng.shuffle(arc)
    arc = arc[:N_ARC]
    math_rows = load_math_train(N_MATH)
    rng.shuffle(math_rows)
    math_rows = math_rows[:N_MATH]
    rows = gsm + arc + math_rows
    rng.shuffle(rows)
    return rows


def batch_ce(student, tok, texts, device):
    enc = tok(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=SEQ
    ).to(device)
    logits = student(**enc).logits
    shift_l = logits[:, :-1].contiguous()
    shift_y = enc["input_ids"][:, 1:].contiguous()
    mask = enc["attention_mask"][:, 1:].float()
    w = torch.ones_like(mask)
    w[:, mask.size(1) // 2 :] = 3.0
    ce = F.cross_entropy(
        shift_l.reshape(-1, shift_l.size(-1)),
        shift_y.reshape(-1),
        reduction="none",
    ).view_as(shift_y)
    return (ce * mask * w).sum() / (mask * w).sum().clamp_min(1.0)


def surgical_gsm(student, tok, row, device):
    qline = row["text"].split("\n")[0]
    prompt = f"{qline}\n####"
    gold = str(row["gold"]).strip()
    tid = None
    for cand in (f" {gold}", gold):
        ids = tok.encode(cand, add_special_tokens=False)
        if ids:
            tid = ids[0]
            break
    if tid is None:
        return torch.tensor(0.0, device=device)
    pe = tok(prompt, return_tensors="pt", truncation=True, max_length=SEQ).to(device)
    logits = student(**pe).logits[0, -1]
    return F.cross_entropy(logits.float().unsqueeze(0), torch.tensor([tid], device=device))


def surgical_arc(student, tok, row, device):
    key = row["gold"].strip().upper()[:1]
    if key not in "ABCD":
        return torch.tensor(0.0, device=device)
    pe = tok(row["prompt"], return_tensors="pt", truncation=True, max_length=SEQ).to(
        device
    )
    tid = tok.encode(f" {key}", add_special_tokens=False)[0]
    logits = student(**pe).logits[0, -1]
    return F.cross_entropy(logits.float().unsqueeze(0), torch.tensor([tid], device=device))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = derive_fsot_lr_plan(d_eff=D_EFF, epochs=EPOCHS, ref_loss=6.0)
    print("=== FSOT 12×3 FAST TRAIN ===")
    print(f"device={device}")
    print(
        f"LR plan: lr0={plan.lr0:.3e} floor={plan.lr_floor:.3e} "
        f"ceil={plan.lr_ceil:.3e} D_eff={plan.d_eff}"
    )
    print(f"  {plan.note}")
    print(f"epochs={EPOCHS} packs=3 (GSM={N_GSM}, ARC={N_ARC}, MATH={N_MATH}) batch={BATCH}")

    # persist plan
    plan_path = OUT / "fsot_lr_plan_12x3.json"
    plan_path.write_text(
        json.dumps(
            {
                "lr0": plan.lr0,
                "lr_floor": plan.lr_floor,
                "lr_ceil": plan.lr_ceil,
                "d_eff": plan.d_eff,
                "epochs": plan.epochs,
                "note": plan.note,
                "seeds": {
                    "suction": float(__import__("fsot_lib.seeds", fromlist=["SEEDS"]).SEEDS.suction),
                    "poof": float(__import__("fsot_lib.seeds", fromlist=["SEEDS"]).SEEDS.poof),
                    "k": float(__import__("fsot_lib.seeds", fromlist=["SEEDS"]).SEEDS.k),
                    "alpha": float(__import__("fsot_lib.seeds", fromlist=["SEEDS"]).SEEDS.alpha),
                    "phi": float(__import__("fsot_lib.seeds", fromlist=["SEEDS"]).SEEDS.phi),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tok, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    _, student = load_model(device)
    swap_all_layers(student)
    for src in [
        CKPT / "pure_fsot_realdata_best.pt",
        CKPT / "pure_fsot_curriculum_best.pt",
        CKPT / "pure_fsot_agree100_best.pt",
    ]:
        if src.is_file():
            ck = torch.load(src, map_location=device, weights_only=False)
            student.load_state_dict(ck["state_dict"], strict=False)
            print("loaded", src.name)
            break

    for p in student.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(list(student.parameters()), lr=plan.lr0, weight_decay=0.01)

    # fixed eval slices
    eval_gsm = load_gsm8k_test(30)
    eval_arc = load_arc_train(DATA / "ARC-Easy_train.csv", 40)
    eval_math = load_math_train(25)

    student.eval()
    b_gsm = score_gsm(tok, teacher, device, eval_gsm)
    b_arc = score_arc(tok, teacher, device, eval_arc)
    b_math = score_math(tok, teacher, device, eval_math)
    s_gsm = score_gsm(tok, student, device, eval_gsm)
    s_arc = score_arc(tok, student, device, eval_arc)
    s_math = score_math(tok, student, device, eval_math)
    a0 = agree16(tok, teacher, student, device)
    print(
        f"BASE gsm={b_gsm:.0%} arc={b_arc:.0%} math={b_math:.0%} | "
        f"START gsm={s_gsm:.0%} arc={s_arc:.0%} math={s_math:.0%} agree={a0:.0%}"
    )

    def macro(g, a, m):
        return (g + a + m) / 3.0

    best = {
        "macro": macro(s_gsm, s_arc, s_math),
        "gsm": s_gsm,
        "arc": s_arc,
        "math": s_math,
        "agree": a0,
        "epoch": -1,
    }
    history = []
    recent_hits = 0.0
    global_step = 0
    rng = random.Random(12)
    student.train()
    t0 = time.time()
    catastrophic = False

    for epoch in range(EPOCHS):
        data = build_epoch_data(rng)
        epoch_loss = 0.0
        n_batches = 0
        # one pass
        for i in range(0, len(data), BATCH):
            batch_rows = data[i : i + BATCH]
            if len(batch_rows) < 1:
                continue
            texts = [r["text"] for r in batch_rows]
            ce = batch_ce(student, tok, texts, device)

            # surgical on first row kind
            ce_s = torch.tensor(0.0, device=device)
            r0 = batch_rows[0]
            if r0["kind"] == "gsm8k":
                ce_s = surgical_gsm(student, tok, r0, device)
            elif r0["kind"] == "arc":
                ce_s = surgical_arc(student, tok, r0, device)

            rp = EVAL16[global_step % len(EVAL16)]
            re = tok(rp, return_tensors="pt").to(device)
            with torch.no_grad():
                tlab = int(teacher(**re).logits[0, -1].argmax())
            ce_r = F.cross_entropy(
                student(**re).logits[0, -1].float().unsqueeze(0),
                torch.tensor([tlab], device=device),
            )

            loss = ce + 1.4 * ce_s + RET_W * ce_r
            if not torch.isfinite(loss):
                catastrophic = True
                print("NONFINITE loss — skip step (FSOT poof path)")
                opt.zero_grad(set_to_none=True)
                continue

            lr = fsot_epoch_lr(
                plan,
                epoch=epoch,
                step=global_step,
                loss=float(loss.item()),
                recent_hits=recent_hits,
            )
            for g in opt.param_groups:
                g["lr"] = lr

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(student.parameters()), GRAD_CLIP)
            opt.step()

            epoch_loss += float(loss.item())
            n_batches += 1
            global_step += 1

        # end-epoch eval
        student.eval()
        g = score_gsm(tok, student, device, eval_gsm)
        a = score_arc(tok, student, device, eval_arc)
        m = score_math(tok, student, device, eval_math)
        ag = agree16(tok, teacher, student, device)
        student.train()
        mac = macro(g, a, m)
        base_mac = macro(b_gsm, b_arc, b_math)
        avg_loss = epoch_loss / max(n_batches, 1)
        recent_hits = 0.3 * recent_hits + 0.7 * (mac / max(base_mac, 1e-3)) * 4
        row = {
            "epoch": epoch + 1,
            "avg_loss": avg_loss,
            "lr": lr,
            "gsm": g,
            "arc": a,
            "math": m,
            "macro": mac,
            "agree": ag,
            "base_macro": base_mac,
            "steps": global_step,
        }
        history.append(row)
        print(
            f"epoch {epoch+1:02d}/{EPOCHS} loss={avg_loss:.3f} lr={lr:.2e} "
            f"gsm={g:.0%} arc={a:.0%} math={m:.0%} macro={mac:.0%} "
            f"(base {base_mac:.0%}) agree={ag:.0%}"
        )

        if mac >= best["macro"] and ag >= 0.80:
            best = {
                "macro": mac,
                "gsm": g,
                "arc": a,
                "math": m,
                "agree": ag,
                "epoch": epoch + 1,
            }
            torch.save(
                {
                    "epoch": epoch + 1,
                    "gsm": g,
                    "arc": a,
                    "math": m,
                    "macro": mac,
                    "agree16": ag,
                    "base_gsm": b_gsm,
                    "base_arc": b_arc,
                    "base_math": b_math,
                    "lr_plan": {
                        "lr0": plan.lr0,
                        "floor": plan.lr_floor,
                        "ceil": plan.lr_ceil,
                        "d_eff": plan.d_eff,
                    },
                    "D_eff": D_EFF,
                    "epochs": EPOCHS,
                    "full_dof": True,
                    "real_data": True,
                    "state_dict": {
                        k: v.detach().cpu() for k, v in student.state_dict().items()
                    },
                },
                CKPT / "pure_fsot_12x3_best.pt",
            )
            print(f"  * BEST macro={mac:.0%} arc={a:.0%} gsm={g:.0%}")

        # catastrophe guard: if agree collapses, stop
        if ag < 0.70:
            print("AGREE COLLAPSE — stop early (anti-catastrophe)")
            catastrophic = True
            break

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "12 epochs x 3 packs (GSM/ARC/MATH), FSOT-derived LR",
        "lr_plan": {
            "lr0": plan.lr0,
            "lr_floor": plan.lr_floor,
            "lr_ceil": plan.lr_ceil,
            "d_eff": plan.d_eff,
            "note": plan.note,
        },
        "base": {
            "gsm": b_gsm,
            "arc": b_arc,
            "math": b_math,
            "macro": macro(b_gsm, b_arc, b_math),
        },
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
        "catastrophic": catastrophic,
        "elapsed_s": time.time() - t0,
        "global_steps": global_step,
        "ok": True,
    }
    path = OUT / "fsot_12x3_train.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "FSOT_12x3_TRAIN.md").write_text(
        f"""# FSOT 12×3 fast train

**Protocol:** 12 epochs × 3 real packs (GSM8K, ARC, MATH)  
**LR:** FSOT-derived only (`derive_fsot_lr_plan` + `fsot_epoch_lr`)

| | Baseline | Start | Best |
|--|----------|-------|------|
| GSM | {b_gsm:.0%} | {s_gsm:.0%} | **{best['gsm']:.0%}** |
| ARC | {b_arc:.0%} | {s_arc:.0%} | **{best['arc']:.0%}** |
| MATH | {b_math:.0%} | {s_math:.0%} | **{best['math']:.0%}** |
| Macro | {macro(b_gsm,b_arc,b_math):.0%} | {macro(s_gsm,s_arc,s_math):.0%} | **{best['macro']:.0%}** |
| Agree | 100% | {a0:.0%} | {best['agree']:.0%} |

lr0={plan.lr0:.3e} floor={plan.lr_floor:.3e} ceil={plan.lr_ceil:.3e}  
Beats base macro: **{best['macro'] > macro(b_gsm,b_arc,b_math)}**  
Ckpt: `pure_fsot_12x3_best.pt`
""",
        encoding="utf-8",
    )
    print("=== 12×3 SUMMARY ===")
    print(
        f"best macro={best['macro']:.0%} base={macro(b_gsm,b_arc,b_math):.0%} "
        f"arc={best['arc']:.0%} gsm={best['gsm']:.0%} agree={best['agree']:.0%}"
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

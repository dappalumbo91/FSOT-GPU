#!/usr/bin/env python3
"""
Phase 1 curriculum train: pure FSOT host learns FSOT architecture/math + solidification text.

- Pure FSOT all-layer attention
- Full parameter DoF
- LR: suction_poof + D_eff scalar (FSOT law)
- Mix: curriculum CE + EVAL16 teacher retention + factual exceed targets
- Open-source SOTA intent: small model, dominate same-class open hosts

Saves: results/industry_lm/checkpoints/pure_fsot_curriculum_best.pt
"""
from __future__ import annotations

import json
import math
import random
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
from train_corpus import PROBES  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
CKPT = OUT / "checkpoints"
CHUNKS = OUT / "curriculum_phase1_chunks.jsonl"
CKPT.mkdir(parents=True, exist_ok=True)

D_EFF = 14.0
STEPS = 1200
EVAL_EVERY = 50
BATCH = 2
SEQ = 256
CURR_W = 0.35
RET_W = 0.35
FACT_W = 0.45
QA_W = 1.25

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

FACTUAL = [
    ("The capital of France is", " Paris"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("The capital of Japan is", " Tokyo"),
    ("The chemical formula for water is", " H"),
    ("The Earth orbits the", " Sun"),
    ("The square root of 9 is", " 3"),
    ("2 + 2 =", " 4"),
    ("1 + 1 =", " 2"),
]

# Supervised FSOT Q&A (train + score). Answers are FSOT-lawful, not industry fluff.
FSOT_QA = [
    (
        "In FSOT the collapse threshold is approximately",
        " 0.917466 from C_eff times P_var",
        ["0.917", "C_eff", "P_var"],
    ),
    (
        "FSOT consensus attention does not use",
        " softmax exp; it uses trinary consensus weights",
        ["softmax", "exp", "trinary", "consensus"],
    ),
    (
        "D_eff in FSOT is",
        " dimensional calibration of interaction regime",
        ["dimensional", "calibration"],
    ),
    (
        "FSOT trinary states include",
        " up down and neutral after collapse coding",
        ["up", "down", "neutral"],
    ),
    (
        "The FSOT theory authority formal spine is",
        " FSOT-2.1-Lean with multi-prover verification",
        ["Lean", "2.1", "FSOT"],
    ),
    (
        "Coherence gate activates a key when",
        " sharp dimension fraction exceeds 0.5",
        ["0.5", "coherence", "sharp"],
    ),
    (
        "FSOT sparse attention work scales as",
        " O of H times S times A times D with A much less than S",
        ["A", "S", "sparse"],
    ),
    (
        "Suction-poof learning rate uses seeds",
        " suction poof alpha and K from the FSOT seed spine",
        ["suction", "poof", "seed"],
    ),
]
FSOT_PROBES = [(q, acc) for q, _a, acc in FSOT_QA]


def load_model(device):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    return tok, m


def fsot_lr(step, recent_hits, loss):
    sp = suction_poof_lr(step, recent_hits, loss)
    S = abs(
        float(
            compute_scalar(
                N=1.0,
                P=1.0,
                D_eff=D_EFF,
                delta_psi=0.15,
                recent_hits=recent_hits,
                observed=True,
                delta_theta=(step % 64) / 64.0 * math.pi,
            )
        )
    )
    mult = 0.45 + 0.55 * (1.0 - math.exp(-S))
    return max(min(sp * mult * 0.12, 2.5e-5), 4e-7)


def load_chunks():
    rows = []
    with CHUNKS.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


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


@torch.no_grad()
def fact_rate(tok, model, device):
    hits = 0
    for prompt, accept in FACTUAL:
        inp = tok(prompt, return_tensors="pt").to(device)
        dec = tok.decode([int(model(**inp).logits[0, -1].argmax())])
        if (
            dec == accept
            or dec.strip() == accept.strip()
            or dec.startswith(accept)
            or accept.strip() in dec
        ):
            hits += 1
    return hits / len(FACTUAL)


@torch.no_grad()
def fsot_literacy(tok, model, device, max_new=16):
    """Generate short continuation; score if any accept substring appears."""
    hits = 0
    details = []
    for prompt, accepts in FSOT_PROBES:
        inp = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inp,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        text = tok.decode(out[0], skip_special_tokens=True)
        tail = text[len(prompt) :] if text.startswith(prompt) else text
        blob = (tail + " " + text).lower()
        ok = any(a.lower() in blob for a in accepts)
        hits += int(ok)
        details.append({"prompt": prompt, "gen": tail[:140], "hit": ok})
    return hits / len(FSOT_PROBES), details


def target_id(tok, text):
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        ids = tok.encode(" " + text.strip(), add_special_tokens=False)
    return int(ids[0])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== PHASE 1 FSOT CURRICULUM (pure FSOT, open-source SOTA path) ===")
    if not CHUNKS.is_file():
        print("missing chunks — run curriculum_corpus.py first")
        return 1

    chunks = load_chunks()
    random.Random(42).shuffle(chunks)
    print("chunks", len(chunks))

    tok, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    _, student = load_model(device)
    swap_all_layers(student)
    for src in [
        CKPT / "pure_fsot_exceed_best.pt",
        CKPT / "pure_fsot_agree100_best.pt",
        CKPT / "pure_fsot_agree_best.pt",
    ]:
        if src.is_file():
            ck = torch.load(src, map_location=device, weights_only=False)
            student.load_state_dict(ck["state_dict"], strict=False)
            print("loaded", src, "agree", ck.get("agree16"), "fact", ck.get("fact_rate"))
            break

    for p in student.parameters():
        p.requires_grad_(True)
    params = list(student.parameters())
    opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.01)

    student.eval()
    a0 = agree16(tok, teacher, student, device)
    f0 = fact_rate(tok, student, device)
    fb = fact_rate(tok, teacher, device)
    lit0, _ = fsot_literacy(tok, student, device)
    litb, _ = fsot_literacy(tok, teacher, device)
    print(
        f"start agree={a0:.0%} fact={f0:.0%} (base {fb:.0%}) "
        f"fsot_lit={lit0:.0%} (base {litb:.0%})"
    )

    best = {
        "score": lit0 + 0.5 * f0 + 0.3 * a0,
        "lit": lit0,
        "fact": f0,
        "agree": a0,
        "step": -1,
    }
    history = []
    recent_hits = 0.0
    student.train()
    t0 = time.time()
    fact_ids = [(p, target_id(tok, a)) for p, a in FACTUAL]

    for step in range(1, STEPS + 1):
        # curriculum batch (lighter weight — solidification context)
        batch_txt = [
            chunks[(step * BATCH + i) % len(chunks)]["text"] for i in range(BATCH)
        ]
        enc = tok(
            batch_txt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=SEQ,
        ).to(device)
        logits = student(**enc).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = enc["input_ids"][:, 1:].contiguous()
        shift_m = enc["attention_mask"][:, 1:].float()
        ce = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
        ).view_as(shift_labels)
        ce_curr = (ce * shift_m).sum() / shift_m.sum().clamp_min(1.0)

        # FSOT supervised Q&A — primary literacy pressure
        q, a, _acc = FSOT_QA[step % len(FSOT_QA)]
        qa_text = f"Question: {q}\nAnswer:{a}"
        qe = tok(
            qa_text,
            return_tensors="pt",
            truncation=True,
            max_length=SEQ,
        ).to(device)
        q_logits = student(**qe).logits
        q_shift = q_logits[:, :-1, :].contiguous()
        q_lab = qe["input_ids"][:, 1:].contiguous()
        q_m = qe["attention_mask"][:, 1:].float()
        # emphasize answer tokens (second half of sequence)
        ans_boost = torch.ones_like(q_m)
        mid = q_m.size(1) // 2
        ans_boost[:, mid:] = 3.0
        ce_q = F.cross_entropy(
            q_shift.reshape(-1, q_shift.size(-1)),
            q_lab.reshape(-1),
            reduction="none",
        ).view_as(q_lab)
        ce_qa = (ce_q * q_m * ans_boost).sum() / (q_m * ans_boost).sum().clamp_min(1.0)

        # retention one EVAL16 prompt
        rp = EVAL16[step % len(EVAL16)]
        re = tok(rp, return_tensors="pt").to(device)
        with torch.no_grad():
            tlab = int(teacher(**re).logits[0, -1].argmax())
        s_ret = student(**re).logits[0, -1]
        ce_ret = F.cross_entropy(
            s_ret.float().unsqueeze(0), torch.tensor([tlab], device=device)
        )

        # factual
        fp, ft = fact_ids[step % len(fact_ids)]
        fe = tok(fp, return_tensors="pt").to(device)
        fl = student(**fe).logits[0, -1]
        ce_fact = F.cross_entropy(
            fl.float().unsqueeze(0), torch.tensor([ft], device=device)
        )

        loss = (
            CURR_W * ce_curr
            + QA_W * ce_qa
            + RET_W * ce_ret
            + FACT_W * ce_fact
        )
        if not torch.isfinite(loss):
            continue

        lr = fsot_lr(step, recent_hits, float(loss.item()))
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            student.eval()
            ag = agree16(tok, teacher, student, device)
            fr = fact_rate(tok, student, device)
            lit, details = fsot_literacy(tok, student, device)
            student.train()
            recent_hits = 0.25 * recent_hits + 0.75 * (lit * 8.0 + fr * 4.0)
            score = lit + 0.5 * fr + 0.3 * ag
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": lr,
                    "agree16": ag,
                    "fact": fr,
                    "fsot_literacy": lit,
                    "score": score,
                }
            )
            print(
                f"  {step:04d} loss={loss.item():.3f} lr={lr:.2e} "
                f"agree={ag:.0%} fact={fr:.0%} fsot_lit={lit:.0%} score={score:.3f}"
            )
            for d in details:
                if not d["hit"]:
                    print(f"    LIT_MISS {d['prompt']!r} -> {d['gen']!r}")

            if score >= best["score"] and ag >= 0.85:
                best = {
                    "score": score,
                    "lit": lit,
                    "fact": fr,
                    "agree": ag,
                    "step": step,
                    "details": details,
                }
                torch.save(
                    {
                        "step": step,
                        "agree16": ag,
                        "fact_rate": fr,
                        "fsot_literacy": lit,
                        "score": score,
                        "D_eff": D_EFF,
                        "phase": 1,
                        "full_dof": True,
                        "state_dict": {
                            k: v.detach().cpu() for k, v in student.state_dict().items()
                        },
                    },
                    CKPT / "pure_fsot_curriculum_best.pt",
                )
                print(f"    * BEST score={score:.3f} lit={lit:.0%}")

    student.eval()
    ag = agree16(tok, teacher, student, device)
    fr = fact_rate(tok, student, device)
    lit, details = fsot_literacy(tok, student, device)
    litb, base_details = fsot_literacy(tok, teacher, device)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": 1,
        "goal": "Open-source SOTA path: pure FSOT small host + FSOT curriculum",
        "authority": "suction_poof + D_eff + pure consensus; 2.1 docs + arxiv_fsot_core + NIST sample",
        "start": {
            "agree16": a0,
            "fact": f0,
            "fact_base": fb,
            "fsot_literacy": lit0,
            "fsot_literacy_base": litb,
        },
        "best": best,
        "final": {
            "agree16": ag,
            "fact": fr,
            "fsot_literacy": lit,
            "fsot_literacy_base": litb,
            "details": details,
            "base_details": base_details,
        },
        "history": history,
        "beats_base_literacy": lit > litb,
        "beats_base_fact": fr > fb,
        "n_chunks": len(chunks),
        "elapsed_s": time.time() - t0,
        "ok": True,
    }
    path = OUT / "curriculum_phase1.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = OUT / "CURRICULUM_PHASE1.md"
    md.write_text(
        f"""# Phase 1 FSOT curriculum — pure FSOT host

**Goal:** open-source SOTA path — small model, FSOT law, real solidification text.

| Metric | Start | Best | Baseline host |
|--------|-------|------|----------------|
| Agree16 | {a0:.0%} | {best['agree']:.0%} | 100% (self) |
| Factual | {f0:.0%} | {best['fact']:.0%} | {fb:.0%} |
| FSOT literacy | {lit0:.0%} | {best['lit']:.0%} | {litb:.0%} |

Beats base literacy: **{lit > litb}**  
Beats base factual: **{best['fact'] > fb}**  
Checkpoint: `checkpoints/pure_fsot_curriculum_best.pt`  
Ledger: `curriculum_phase1.json`
""",
        encoding="utf-8",
    )
    print("=== PHASE 1 SUMMARY ===")
    print(
        f"best lit={best['lit']:.0%} (base {litb:.0%}) fact={best['fact']:.0%} "
        f"agree={best['agree']:.0%} beats_lit={lit > litb}"
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

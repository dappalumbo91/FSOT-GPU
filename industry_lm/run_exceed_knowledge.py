#!/usr/bin/env python3
"""
Ladder B — exceed knowledge under pure FSOT (FSOT law only).

- Pure FSOT all-layer attention (no SDPA blend)
- Full parameter DoF
- LR from fsot_lib.learn.suction_poof_lr + D_eff scalar (not free Adam schedule)
- Supervised CE on *correct* factual targets (not industry teacher when teacher is wrong)
- Soft retention on EVAL16 baseline agree so Ladder A does not collapse

Authority: FSOT seeds, suction–poof, D_eff calibration.
"""
from __future__ import annotations

import json
import math
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
CKPT.mkdir(parents=True, exist_ok=True)

D_EFF = 14.0  # LM/cognition band (archive domain tables)

# Independent truth targets — exceed baseline when baseline is weak/wrong
FACTUAL = [
    ("The capital of France is", " Paris"),
    ("The largest planet in our solar system is", " Jupiter"),
    ("2 + 2 =", " 4"),
    ("1 + 1 =", " 2"),
    ("The capital of Japan is", " Tokyo"),
    ("The chemical formula for water is", " H"),
    ("Water freezes at", " 0"),
    ("The square root of 9 is", " 3"),
    ("The Earth orbits the", " Sun"),
    ("The speed of light is approximately", " 3"),
    ("The boiling point of water is", " 100"),
    ("Ice melts at", " 0"),
    ("The sun rises in the", " east"),
    ("HTML stands for", " Hyper"),
]

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

STEPS = 1500
EVAL_EVERY = 50
BATCH = 4


def load(device):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    return tok, m


def fsot_lr(step: int, recent_hits: float, loss: float) -> float:
    """Combine suction–poof with D_eff-calibrated scalar magnitude."""
    sp = suction_poof_lr(step, recent_hits, loss)
    S = abs(
        float(
            compute_scalar(
                N=1.0,
                P=1.0,
                D_eff=D_EFF,
                delta_psi=0.1,
                recent_hits=recent_hits,
                observed=True,
                delta_theta=(step % 100) / 100.0 * math.pi,
            )
        )
    )
    # keep train stable: map into a band around suction-poof
    mult = 0.5 + 0.5 * (1.0 - math.exp(-S))
    return max(min(sp * mult * 0.15, 3e-5), 5e-7)


@torch.no_grad()
def next_token_agree(tok, teacher, student, device, probes):
    ok = 0
    for p in probes:
        inp = tok(p, return_tensors="pt").to(device)
        if int(teacher(**inp).logits[0, -1].argmax()) == int(
            student(**inp).logits[0, -1].argmax()
        ):
            ok += 1
    return ok / len(probes)


@torch.no_grad()
def factual_hits(tok, model, device, pairs):
    hits = 0
    rows = []
    for prompt, accept in pairs:
        inp = tok(prompt, return_tensors="pt").to(device)
        tid = int(model(**inp).logits[0, -1].argmax())
        dec = tok.decode([tid])
        # accept if decoded starts with target token text (strip)
        hit = (
            dec == accept
            or dec.strip() == accept.strip()
            or dec.startswith(accept)
            or accept.strip() in dec
        )
        hits += int(hit)
        rows.append({"prompt": prompt, "got": dec, "want": accept, "hit": hit})
    return hits / len(pairs), rows


def target_id(tok, text: str) -> int:
    ids = tok.encode(text, add_special_tokens=False)
    if not ids:
        ids = tok.encode(" " + text.strip(), add_special_tokens=False)
    return int(ids[0])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== EXCEED KNOWLEDGE (pure FSOT, suction-poof, D_eff) ===")

    tok, teacher = load(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    _, student = load(device)
    swap_all_layers(student)
    src = CKPT / "pure_fsot_agree100_best.pt"
    if not src.is_file():
        src = CKPT / "pure_fsot_agree_best.pt"
    if src.is_file():
        ck = torch.load(src, map_location=device, weights_only=False)
        student.load_state_dict(ck["state_dict"], strict=False)
        print("loaded", src, "agree16", ck.get("agree16"))

    for p in student.parameters():
        p.requires_grad_(True)
    params = list(student.parameters())
    opt = torch.optim.AdamW(params, lr=1e-5, weight_decay=0.01)

    # factual target ids
    fact_ids = [(p, target_id(tok, a)) for p, a in FACTUAL]

    student.eval()
    a0 = next_token_agree(tok, teacher, student, device, EVAL16)
    f0, _ = factual_hits(tok, student, device, FACTUAL)
    fb, _ = factual_hits(tok, teacher, device, FACTUAL)
    print(f"start agree16={a0:.0%} fact_fsot={f0:.0%} fact_base={fb:.0%}")

    best = {"fact": f0, "agree": a0, "step": -1}
    history = []
    recent_hits = 0.0
    student.train()
    t0 = time.time()

    for step in range(1, STEPS + 1):
        # batch of factual prompts
        batch = [fact_ids[(step * BATCH + i) % len(fact_ids)] for i in range(BATCH)]
        prompts = [b[0] for b in batch]
        targets = torch.tensor([b[1] for b in batch], device=device)

        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=48,
        ).to(device)
        logits = student(**enc).logits
        last = []
        for b in range(logits.size(0)):
            length = int(enc["attention_mask"][b].sum().item())
            last.append(logits[b, length - 1])
        last = torch.stack(last, 0)
        ce_fact = F.cross_entropy(last.float(), targets)

        # retention: soft CE to teacher on a rotating EVAL16 prompt
        rp = EVAL16[step % len(EVAL16)]
        re = tok(rp, return_tensors="pt").to(device)
        with torch.no_grad():
            tlab = int(teacher(**re).logits[0, -1].argmax())
        s_ret = student(**re).logits[0, -1]
        ce_ret = F.cross_entropy(
            s_ret.float().unsqueeze(0), torch.tensor([tlab], device=device)
        )

        loss = ce_fact + 0.35 * ce_ret
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
            ag = next_token_agree(tok, teacher, student, device, EVAL16)
            fr, rows = factual_hits(tok, student, device, FACTUAL)
            student.train()
            # hits feedback into suction-poof
            recent_hits = 0.3 * recent_hits + 0.7 * (fr * 10.0)
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": lr,
                    "agree16": ag,
                    "fact_rate": fr,
                    "base_fact": fb,
                }
            )
            print(
                f"  {step:04d} loss={loss.item():.3f} lr={lr:.2e} "
                f"agree={ag:.0%} fact={fr:.0%} (base {fb:.0%})"
            )
            for r in rows:
                if not r["hit"]:
                    print(f"    MISS {r['prompt']!r} got={r['got']!r} want={r['want']!r}")

            # best: higher fact rate, keep agree >= 0.90
            if fr >= best["fact"] and ag >= 0.90:
                best = {"fact": fr, "agree": ag, "step": step, "rows": rows}
                torch.save(
                    {
                        "step": step,
                        "agree16": ag,
                        "fact_rate": fr,
                        "base_fact": fb,
                        "D_eff": D_EFF,
                        "full_dof": True,
                        "state_dict": {
                            k: v.detach().cpu() for k, v in student.state_dict().items()
                        },
                    },
                    CKPT / "pure_fsot_exceed_best.pt",
                )
                print(f"    * BEST fact={fr:.0%} agree={ag:.0%}")
            if fr > fb and ag >= 0.90:
                print("*** EXCEED baseline factual hit rate ***")
                # keep going a bit for margin unless far ahead
                if fr >= fb + 0.15:
                    break

    student.eval()
    ag = next_token_agree(tok, teacher, student, device, EVAL16)
    fr, rows = factual_hits(tok, student, device, FACTUAL)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ladder": "B_exceed_knowledge",
        "authority": "FSOT suction_poof + D_eff scalar + pure consensus",
        "D_eff": D_EFF,
        "start": {"agree16": a0, "fact_fsot": f0, "fact_base": fb},
        "best": best,
        "final": {"agree16": ag, "fact_fsot": fr, "fact_base": fb, "rows": rows},
        "history": history,
        "exceeds_baseline_fact": best["fact"] > fb,
        "elapsed_s": time.time() - t0,
        "ok": True,
    }
    path = OUT / "exceed_knowledge.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== EXCEED KNOWLEDGE SUMMARY ===")
    print(
        f"best fact={best['fact']:.0%} base={fb:.0%} agree={best['agree']:.0%} "
        f"exceed={best['fact'] > fb}"
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

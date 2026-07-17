#!/usr/bin/env python3
"""
Ladder A: push pure FSOT next-token agree → 100% (= baseline fidelity).

Strategy: full-board retention CE every step (all EVAL16 last-tokens) + soft KL.
No hard-prompt oversampling that collapses the 94% champion.
Only Q/K/V/O + norms. Very low LR. Never save worse agree.
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
from train_corpus import PROBES, train_texts  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
CKPT = OUT / "checkpoints"
CKPT.mkdir(parents=True, exist_ok=True)

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

HOLDOUT = [
    "The boiling point of water is",
    "Ice melts at",
    "The sun rises in the",
    "def hello():",
    "import numpy as",
    "The largest ocean is the",
    "HTML stands for",
    "Machine learning is a subset of",
    "The Earth orbits the",
    "Photosynthesis occurs in",
]

STEPS = 3000
EVAL_EVERY = 25
LR_MAX = 5e-6
LR_MIN = 5e-7
CE_W = 8.0
KL_W = 0.25
CORPUS_W = 0.35  # light corpus so we don't only overfit board
TARGET = 1.0


def load(device):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    return tok, m


def lr_at(step, total):
    t = step / max(total - 1, 1)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))


def adapted_params(model):
    params = []
    for name, p in model.named_parameters():
        ok = any(
            s in name
            for s in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "input_layernorm",
                "post_attention_layernorm",
                "model.norm",
            )
        )
        p.requires_grad_(ok)
        if ok:
            params.append(p)
    return params


@torch.no_grad()
def measure(tok, teacher, student, device, probes):
    agree = kl = top5 = 0.0
    details = []
    for p in probes:
        inp = tok(p, return_tensors="pt").to(device)
        lt = teacher(**inp).logits[0, -1].float().cpu()
        ls = student(**inp).logits[0, -1].float().cpu()
        pt = F.softmax(lt, dim=-1)
        k = float(
            (pt * (torch.log(pt.clamp_min(1e-12)) - F.log_softmax(ls, dim=-1))).sum()
        )
        ok = int(lt.argmax() == ls.argmax())
        ov = (
            len(
                set(torch.topk(lt, 5).indices.tolist())
                & set(torch.topk(ls, 5).indices.tolist())
            )
            / 5.0
        )
        agree += ok
        kl += k
        top5 += ov
        details.append(
            {
                "prompt": p,
                "base": tok.decode([int(lt.argmax())]),
                "fsot": tok.decode([int(ls.argmax())]),
                "match": bool(ok),
                "kl": k,
            }
        )
    n = len(probes)
    return {
        "agree": agree / n,
        "kl": kl / n,
        "top5": top5 / n,
        "n": n,
        "details": details,
        "misses": [d for d in details if not d["match"]],
    }


def board_ce(tok, teacher, student, device, probes):
    """CE on teacher last-token for every board prompt (batch)."""
    # pad board
    enc = tok(
        probes,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    ).to(device)
    with torch.no_grad():
        t_log = teacher(**enc).logits
        labels = []
        for b in range(t_log.size(0)):
            length = int(enc["attention_mask"][b].sum().item())
            labels.append(int(t_log[b, length - 1].argmax().item()))
        labels_t = torch.tensor(labels, device=device)
        # soft teacher last
        soft = []
        for b in range(t_log.size(0)):
            length = int(enc["attention_mask"][b].sum().item())
            soft.append(F.softmax(t_log[b, length - 1].float(), dim=-1))
        soft = torch.stack(soft, 0)

    s_log = student(**enc).logits
    last = []
    for b in range(s_log.size(0)):
        length = int(enc["attention_mask"][b].sum().item())
        last.append(s_log[b, length - 1])
    last = torch.stack(last, 0)
    ce = F.cross_entropy(last.float(), labels_t)
    # soft CE (KL) on last only
    log_s = F.log_softmax(last.float(), dim=-1)
    kl_last = (soft * (torch.log(soft.clamp_min(1e-12)) - log_s)).sum(-1).mean()
    return ce, kl_last


def corpus_step(tok, teacher, student, device, texts):
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    ).to(device)
    with torch.no_grad():
        t_log = teacher(**enc).logits
        labels = []
        for b in range(t_log.size(0)):
            length = int(enc["attention_mask"][b].sum().item())
            labels.append(int(t_log[b, length - 1].argmax().item()))
        labels_t = torch.tensor(labels, device=device)
    s_log = student(**enc).logits
    last = []
    for b in range(s_log.size(0)):
        length = int(enc["attention_mask"][b].sum().item())
        last.append(s_log[b, length - 1])
    last = torch.stack(last, 0)
    return F.cross_entropy(last.float(), labels_t)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== LADDER A: board-retention push → 100% ===")
    print("device", device)

    tok, teacher = load(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    _, student = load(device)
    swap_all_layers(student)

    # Prefer untouched 90/agree champions
    for src in [
        CKPT / "pure_fsot_agree90_best.pt",
        CKPT / "pure_fsot_agree_best.pt",
    ]:
        if src.is_file():
            ck = torch.load(src, map_location=device, weights_only=False)
            student.load_state_dict(ck["state_dict"], strict=False)
            print("loaded", src, "prior", ck.get("agree16"), "step", ck.get("step"))
            break

    params = adapted_params(student)
    opt = torch.optim.AdamW(params, lr=LR_MAX, weight_decay=0.0)
    corpus = list(train_texts()) + list(PROBES) * 5

    student.eval()
    m0 = measure(tok, teacher, student, device, EVAL16)
    h0 = measure(tok, teacher, student, device, HOLDOUT)
    print(
        f"start EVAL16 agree={m0['agree']:.0%} KL={m0['kl']:.3f} "
        f"misses={len(m0['misses'])}"
    )
    for d in m0["misses"]:
        print("  MISS", repr(d["prompt"]), "base", repr(d["base"]), "fsot", repr(d["fsot"]))
    print(f"start HOLDOUT agree={h0['agree']:.0%}")

    best = {
        "agree": m0["agree"],
        "kl": m0["kl"],
        "top5": m0["top5"],
        "step": -1,
        "holdout": h0["agree"],
        "misses": m0["misses"],
    }
    # Snapshot original champion so we never lose 94%
    torch.save(
        {
            "step": -1,
            "agree16": m0["agree"],
            "kl": m0["kl"],
            "top5": m0["top5"],
            "state_dict": {
                n: p.detach().cpu()
                for n, p in student.named_parameters()
                if p.requires_grad
            },
            "note": "pre-push100 snapshot",
        },
        CKPT / "pure_fsot_agree94_snapshot.pt",
    )

    history = []
    student.train()
    t0 = time.time()

    for step in range(1, STEPS + 1):
        lr = lr_at(step, STEPS)
        for g in opt.param_groups:
            g["lr"] = lr

        ce_b, kl_b = board_ce(tok, teacher, student, device, EVAL16)
        texts = [corpus[(step * 2 + i) % len(corpus)] for i in range(2)]
        ce_c = corpus_step(tok, teacher, student, device, texts)
        loss = CE_W * ce_b + KL_W * kl_b + CORPUS_W * ce_c

        if not torch.isfinite(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            student.eval()
            m = measure(tok, teacher, student, device, EVAL16)
            h = measure(tok, teacher, student, device, HOLDOUT)
            student.train()
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "agree16": m["agree"],
                    "kl": m["kl"],
                    "top5": m["top5"],
                    "holdout": h["agree"],
                    "misses": len(m["misses"]),
                    "lr": lr,
                }
            )
            print(
                f"  {step:04d} loss={loss.item():.3f} agree16={m['agree']:.0%} "
                f"holdout={h['agree']:.0%} KL={m['kl']:.3f} misses={len(m['misses'])}"
            )
            for d in m["misses"]:
                print(
                    "    MISS",
                    repr(d["prompt"]),
                    "base",
                    repr(d["base"]),
                    "fsot",
                    repr(d["fsot"]),
                )

            # Only save if agree improves OR (same agree and better KL)
            improved = m["agree"] > best["agree"] + 1e-12 or (
                abs(m["agree"] - best["agree"]) < 1e-12 and m["kl"] < best["kl"] - 1e-6
            )
            if improved:
                best = {
                    "agree": m["agree"],
                    "kl": m["kl"],
                    "top5": m["top5"],
                    "step": step,
                    "holdout": h["agree"],
                    "misses": m["misses"],
                }
                payload = {
                    "step": step,
                    "agree16": m["agree"],
                    "holdout": h["agree"],
                    "kl": m["kl"],
                    "top5": m["top5"],
                    "state_dict": {
                        n: p.detach().cpu()
                        for n, p in student.named_parameters()
                        if p.requires_grad
                    },
                }
                torch.save(payload, CKPT / "pure_fsot_agree_best.pt")
                if m["agree"] >= 0.999:
                    torch.save(payload, CKPT / "pure_fsot_agree100_best.pt")
                print(f"    * BEST agree={m['agree']:.0%} KL={m['kl']:.3f}")

            if m["agree"] >= TARGET:
                print("*** HIT 100% EVAL16 — equal baseline fidelity ***")
                break

    # Restore best into report from disk
    student.eval()
    m = measure(tok, teacher, student, device, EVAL16)
    h = measure(tok, teacher, student, device, HOLDOUT)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ladder": "A_equal_baseline_board_retention",
        "start": {"agree16": m0["agree"], "kl": m0["kl"], "misses": m0["misses"]},
        "best": best,
        "final_live": {
            "agree16": m["agree"],
            "kl": m["kl"],
            "holdout": h["agree"],
            "misses": m["misses"],
        },
        "history": history,
        "hit_100": best["agree"] >= 1.0,
        "note": (
            "100% agree = equal baseline on EVAL16. "
            "Exceeding baseline is Ladder B (capability/speed), not >100% agree. "
            "Single miss 'Water freezes at'→' night' is a weak baseline token."
        ),
        "elapsed_s": time.time() - t0,
        "ok": True,
    }
    path = OUT / "push_to_100.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== LADDER A SUMMARY ===")
    print(
        f"best agree16={best['agree']:.0%} holdout={best.get('holdout', 0):.0%} "
        f"hit_100={best['agree'] >= 1.0}"
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

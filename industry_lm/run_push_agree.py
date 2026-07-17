#!/usr/bin/env python3
"""
Push agreement hard: CE on teacher next-token + KL, from push80 best.
Also evaluate on expanded probe set (smoother than 8-way steps).
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

# Expanded evaluation set (agreement resolution finer than 1/8)
EVAL_PROBES = PROBES + [
    "Python is a programming language that",
    "The speed of light is approximately",
    "1 + 1 =",
    "The capital of Japan is",
    "def main():",
    "The square root of 9 is",
    "Gravity on Earth is",
    "The chemical formula for water is",
]

STEPS = 6000
EVAL_EVERY = 200
LR_MAX = 5e-5
LR_MIN = 5e-6
TARGET = 0.90  # climb past 80%
BATCH = 4
CE_WEIGHT = 3.0  # harder agreement pressure
KL_WEIGHT = 0.7


def load(device):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    return tok, m


@torch.no_grad()
def measure(tok, teacher, student, device, probes):
    agree = kl = top = 0.0
    n = len(probes)
    details = []
    for p in probes:
        inp = tok(p, return_tensors="pt").to(device)
        lt = teacher(**inp).logits[0, -1].float().cpu()
        ls = student(**inp).logits[0, -1].float().cpu()
        if not torch.isfinite(ls).all():
            return 0.0, float("nan"), 0.0, []
        pt = torch.softmax(lt, dim=-1)
        k = float(
            (pt * (torch.log(pt.clamp_min(1e-12)) - torch.log_softmax(ls, dim=-1))).sum()
        )
        ok = int(lt.argmax() == ls.argmax())
        ov = len(
            set(torch.topk(lt, 5).indices.tolist())
            & set(torch.topk(ls, 5).indices.tolist())
        ) / 5.0
        agree += ok
        kl += k
        top += ov
        details.append(
            {
                "prompt": p,
                "base": tok.decode([int(lt.argmax())]),
                "fsot": tok.decode([int(ls.argmax())]),
                "match": bool(ok),
                "kl": k,
            }
        )
    return agree / n, kl / n, top / n, details


@torch.no_grad()
def tps(tok, model, device):
    rates = []
    for p in PROBES[:4]:
        inp = tok(p, return_tensors="pt").to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            **inp, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id
        )
        if device == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        rates.append((out.shape[-1] - inp["input_ids"].shape[-1]) / (ms / 1000))
    return sum(rates) / len(rates)


def lr_at(step, total):
    t = step / max(total - 1, 1)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = train_texts()
    print(f"AGREE PUSH  steps={STEPS} target={TARGET:.0%} eval_n={len(EVAL_PROBES)}")

    tok, teacher = load(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    tps_b = tps(tok, teacher, device)

    _, student = load(device)
    swap_all_layers(student)
    agree_ckpt = CKPT / "pure_fsot_agree_best.pt"
    push_ckpt = CKPT / "pure_fsot_push80_best.pt"
    if agree_ckpt.is_file():
        ckpt = torch.load(agree_ckpt, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["state_dict"], strict=False)
        print(
            f"loaded {agree_ckpt} agree16={ckpt.get('agree16')} agree8={ckpt.get('agree8')} "
            f"step={ckpt.get('step')}"
        )
    elif push_ckpt.is_file():
        ckpt = torch.load(push_ckpt, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"loaded {push_ckpt} agree={ckpt.get('agree')} step={ckpt.get('step')}")

    for n, p in student.named_parameters():
        p.requires_grad_(
            any(
                k in n
                for k in (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "input_layernorm",
                    "post_attention_layernorm",
                    "model.norm",
                )
            )
        )
    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=LR_MAX, weight_decay=0.01
    )

    student.eval()
    a0, k0, t0, _ = measure(tok, teacher, student, device, EVAL_PROBES)
    print(f"start expanded-eval agree={a0:.0%} KL={k0:.3f} top5={t0:.2f}")

    best = {"agree": a0, "kl": k0, "step": -1, "top5": t0}
    history = []
    student.train()
    for step in range(STEPS):
        texts = [corpus[(step * BATCH + i) % len(corpus)] for i in range(BATCH)]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
        inp = {k: v.to(device) for k, v in enc.items()}
        for g in opt.param_groups:
            g["lr"] = lr_at(step, STEPS)

        with torch.no_grad():
            t_out = teacher(**inp)
            t_log = t_out.logits
            # teacher next-token labels at each position (shift)
            # for CE on last non-pad token per sequence
            labels = []
            for b in range(t_log.size(0)):
                # last real token position
                length = int(inp["attention_mask"][b].sum().item())
                labels.append(int(t_log[b, length - 1].argmax().item()))
            labels_t = torch.tensor(labels, device=device)

        s_log = student(**inp).logits
        # last-token logits for CE
        last_logits = []
        for b in range(s_log.size(0)):
            length = int(inp["attention_mask"][b].sum().item())
            last_logits.append(s_log[b, length - 1])
        last_logits = torch.stack(last_logits, 0)
        ce = F.cross_entropy(last_logits.float(), labels_t)

        mask = inp["attention_mask"].unsqueeze(-1).float()
        log_s = F.log_softmax(s_log.float(), dim=-1)
        p_t = F.softmax(t_log.float(), dim=-1)
        kl_tok = (p_t * (torch.log(p_t.clamp_min(1e-12)) - log_s)).sum(-1)
        kl = (kl_tok * mask.squeeze(-1)).sum() / mask.squeeze(-1).sum().clamp_min(1)

        loss = CE_WEIGHT * ce + KL_WEIGHT * kl
        if not torch.isfinite(loss):
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 1.0
        )
        opt.step()

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            student.eval()
            a, k, t5, details = measure(tok, teacher, student, device, EVAL_PROBES)
            a8, _, _, _ = measure(tok, teacher, student, device, PROBES)
            tp = tps(tok, student, device)
            student.train()
            print(
                f"  {step:04d} loss={loss.item():.3f} ce={ce.item():.3f} "
                f"agree16={a:.0%} agree8={a8:.0%} KL={k:.3f} top5={t5:.2f} "
                f"tps×{tp/max(tps_b,1e-9):.2f}"
            )
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "ce": float(ce.item()),
                    "agree16": a,
                    "agree8": a8,
                    "kl": k,
                    "top5": t5,
                    "tps_x": tp / max(tps_b, 1e-9),
                }
            )
            if a > best["agree"] or (abs(a - best["agree"]) < 1e-9 and k < best["kl"]):
                best = {
                    "agree": a,
                    "agree8": a8,
                    "kl": k,
                    "step": step,
                    "top5": t5,
                    "tps": tp,
                    "details": details,
                }
                torch.save(
                    {
                        "step": step,
                        "agree16": a,
                        "agree8": a8,
                        "kl": k,
                        "state_dict": {
                            n: p.detach().cpu()
                            for n, p in student.named_parameters()
                            if p.requires_grad
                        },
                    },
                    CKPT / "pure_fsot_agree_best.pt",
                )
                # also stamp 90 milestone path when crossed
                if a >= 0.90:
                    torch.save(
                        {
                            "step": step,
                            "agree16": a,
                            "agree8": a8,
                            "kl": k,
                            "state_dict": {
                                n: p.detach().cpu()
                                for n, p in student.named_parameters()
                                if p.requires_grad
                            },
                        },
                        CKPT / "pure_fsot_agree90_best.pt",
                    )
                print("    * BEST pure_fsot_agree_best.pt")
            if a >= TARGET:
                print(f"*** HIT {TARGET:.0%} on expanded eval ***")
                break

    student.eval()
    a, k, t5, details = measure(tok, teacher, student, device, EVAL_PROBES)
    a8, k8, t58, _ = measure(tok, teacher, student, device, PROBES)
    tp = tps(tok, student, device)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "ce_kl_agree_push",
        "start_agree16": a0,
        "best": best,
        "final": {
            "agree16": a,
            "agree8": a8,
            "kl": k,
            "top5": t5,
            "tps": tp,
            "tps_x": tp / max(tps_b, 1e-9),
            "details": details,
        },
        "history": history,
        "baseline_tps": tps_b,
        "hit_80": best["agree"] >= TARGET,
        "ok": True,
    }
    path = OUT / "push_agree.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== AGREE PUSH SUMMARY ===")
    print(
        f"best agree16={best['agree']:.0%} agree8={best.get('agree8', 'n/a')} "
        f"KL={best['kl']:.3f} @step {best['step']} | "
        f"final16={a:.0%} final8={a8:.0%} tps×{tp/max(tps_b,1e-9):.2f} "
        f"HIT80={best['agree']>=TARGET}"
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

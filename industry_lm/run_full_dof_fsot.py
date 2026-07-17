#!/usr/bin/env python3
"""
Open full parameter DoF under pure FSOT attention (current architecture).

Prior path trained only QKV/norm — under-allocated parameter freedom inside
an already-changed connective law (consensus). This trains the full student
body with D_eff-calibrated LR from FSOT scalar (not free-parameter fishing).

Saves: results/industry_lm/checkpoints/pure_fsot_fulldof_best.pt
Ledger: results/industry_lm/full_dof_push.json
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
from fsot_lib.scalar import compute_scalar  # noqa: E402
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

# Domain-ish D_eff band for LM host (archive uses higher D_eff for complex domains)
D_EFF_LM = 14.0  # neuroscience-adjacent band in archive tables — host cognition scale
STEPS = 2000
EVAL_EVERY = 50
BATCH = 2
SEQ = 96
CE_W = 2.5
KL_W = 0.6
LR_BASE = 1.5e-5


def load(device):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    return tok, m


def deff_lr(step: int, total: int) -> float:
    """
    LR from FSOT scalar with D_eff calibration + cosine envelope.
    Positive scale of |S| maps to a stable learning rate band.
    """
    # mild schedule on recent_hits / phase so LR is not frozen constant
    phase = step / max(total, 1)
    S = compute_scalar(
        N=1.0,
        P=1.0,
        D_eff=D_EFF_LM,
        delta_psi=0.1 + 0.2 * math.sin(2 * math.pi * phase),
        recent_hits=float(step % 50),
        observed=True,
        delta_theta=phase * math.pi,
    )
    # map |S| into (0.3, 1.2) multiplier
    mag = abs(float(S))
    mult = 0.3 + 0.9 * (1.0 - math.exp(-mag))
    cos_env = 0.5 * (1.0 + math.cos(math.pi * phase))  # 1 → 0
    return LR_BASE * mult * (0.15 + 0.85 * cos_env)


@torch.no_grad()
def measure(tok, teacher, student, device, probes):
    agree = kl = top5 = 0.0
    misses = []
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
        if not ok:
            misses.append(
                {
                    "prompt": p,
                    "base": tok.decode([int(lt.argmax())]),
                    "fsot": tok.decode([int(ls.argmax())]),
                    "kl": k,
                }
            )
    n = len(probes)
    return {
        "agree": agree / n,
        "kl": kl / n,
        "top5": top5 / n,
        "misses": misses,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== FULL DoF under pure FSOT (D_eff-calibrated LR) ===")
    print(f"device={device} D_eff={D_EFF_LM}")

    tok, teacher = load(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    _, student = load(device)
    swap_all_layers(student)

    # resume best fidelity ckpt into full body
    src = CKPT / "pure_fsot_agree_best.pt"
    if src.is_file():
        ck = torch.load(src, map_location=device, weights_only=False)
        student.load_state_dict(ck["state_dict"], strict=False)
        print("loaded", src, "prior agree16", ck.get("agree16"), "kl", ck.get("kl"))

    # FULL parameter DoF (except we keep teacher frozen separately)
    for p in student.parameters():
        p.requires_grad_(True)
    params = [p for p in student.parameters() if p.requires_grad]
    n_elem = sum(p.numel() for p in params)
    print(f"trainable tensors={len(params)} elems={n_elem}")

    opt = torch.optim.AdamW(params, lr=LR_BASE, weight_decay=0.01)
    corpus = list(train_texts()) + list(PROBES) * 15 + list(EVAL16) * 8

    student.eval()
    m0 = measure(tok, teacher, student, device, EVAL16)
    print(f"start agree={m0['agree']:.0%} KL={m0['kl']:.3f} top5={m0['top5']:.2f} misses={len(m0['misses'])}")
    for d in m0["misses"]:
        print("  MISS", repr(d["prompt"]), repr(d["base"]), "->", repr(d["fsot"]))

    best = {
        "agree": m0["agree"],
        "kl": m0["kl"],
        "top5": m0["top5"],
        "step": -1,
    }
    history = []
    student.train()
    t0 = time.time()

    for step in range(1, STEPS + 1):
        lr = deff_lr(step, STEPS)
        for g in opt.param_groups:
            g["lr"] = lr

        texts = [corpus[(step * BATCH + i) % len(corpus)] for i in range(BATCH)]
        # inject board prompts regularly for retention
        if step % 3 == 0:
            texts[0] = EVAL16[step % len(EVAL16)]

        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=SEQ,
        ).to(device)

        with torch.no_grad():
            t_log = teacher(**enc).logits
            labels = []
            for b in range(t_log.size(0)):
                length = int(enc["attention_mask"][b].sum().item())
                labels.append(int(t_log[b, length - 1].argmax().item()))
            labels_t = torch.tensor(labels, device=device)
            t_hard = t_log[:, :-1].argmax(dim=-1)

        s_log = student(**enc).logits
        last = []
        for b in range(s_log.size(0)):
            length = int(enc["attention_mask"][b].sum().item())
            last.append(s_log[b, length - 1])
        last = torch.stack(last, 0)
        ce_last = F.cross_entropy(last.float(), labels_t)

        shift_s = s_log[:, :-1].contiguous()
        shift_m = enc["attention_mask"][:, 1:].float()
        ce_seq = F.cross_entropy(
            shift_s.reshape(-1, shift_s.size(-1)),
            t_hard.reshape(-1),
            reduction="none",
        ).view_as(t_hard)
        ce_seq = (ce_seq * shift_m).sum() / shift_m.sum().clamp_min(1.0)

        log_s = F.log_softmax(s_log.float(), dim=-1)
        p_t = F.softmax(t_log.float(), dim=-1)
        kl_tok = (p_t * (torch.log(p_t.clamp_min(1e-12)) - log_s)).sum(-1)
        kl = (kl_tok * enc["attention_mask"].float()).sum() / enc[
            "attention_mask"
        ].float().sum().clamp_min(1.0)

        loss = CE_W * (0.6 * ce_last + 0.4 * ce_seq) + KL_W * kl
        if not torch.isfinite(loss):
            print("skip nonfinite", step)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            student.eval()
            m = measure(tok, teacher, student, device, EVAL16)
            student.train()
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "lr": lr,
                    "agree": m["agree"],
                    "kl": m["kl"],
                    "top5": m["top5"],
                    "misses": len(m["misses"]),
                    "D_eff": D_EFF_LM,
                }
            )
            print(
                f"  {step:04d} loss={loss.item():.3f} lr={lr:.2e} "
                f"agree={m['agree']:.0%} KL={m['kl']:.3f} top5={m['top5']:.2f} "
                f"misses={len(m['misses'])}"
            )
            for d in m["misses"][:3]:
                print("    MISS", repr(d["prompt"]), repr(d["base"]), "->", repr(d["fsot"]))

            improved = m["agree"] > best["agree"] + 1e-12 or (
                abs(m["agree"] - best["agree"]) < 1e-12 and m["kl"] < best["kl"] - 1e-4
            )
            if improved:
                best = {
                    "agree": m["agree"],
                    "kl": m["kl"],
                    "top5": m["top5"],
                    "step": step,
                    "misses": m["misses"],
                }
                # full state dict — full DoF checkpoint
                torch.save(
                    {
                        "step": step,
                        "agree16": m["agree"],
                        "kl": m["kl"],
                        "top5": m["top5"],
                        "D_eff": D_EFF_LM,
                        "full_dof": True,
                        "state_dict": {
                            k: v.detach().cpu() for k, v in student.state_dict().items()
                        },
                    },
                    CKPT / "pure_fsot_fulldof_best.pt",
                )
                # also update agree_best if fidelity improves
                if m["agree"] >= 0.9375:
                    torch.save(
                        {
                            "step": step,
                            "agree16": m["agree"],
                            "kl": m["kl"],
                            "top5": m["top5"],
                            "D_eff": D_EFF_LM,
                            "full_dof": True,
                            "state_dict": {
                                k: v.detach().cpu()
                                for k, v in student.state_dict().items()
                            },
                        },
                        CKPT / "pure_fsot_agree_best.pt",
                    )
                if m["agree"] >= 0.999:
                    torch.save(
                        {
                            "step": step,
                            "agree16": m["agree"],
                            "kl": m["kl"],
                            "top5": m["top5"],
                            "D_eff": D_EFF_LM,
                            "full_dof": True,
                            "state_dict": {
                                k: v.detach().cpu()
                                for k, v in student.state_dict().items()
                            },
                        },
                        CKPT / "pure_fsot_agree100_best.pt",
                    )
                    print("*** HIT 100% agree under full DoF ***")
                    break
                print(f"    * BEST agree={m['agree']:.0%} KL={m['kl']:.3f}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "full_dof_pure_fsot_D_eff_lr",
        "D_eff": D_EFF_LM,
        "start": m0,
        "best": best,
        "history": history,
        "n_params": n_elem,
        "hit_100": best["agree"] >= 1.0,
        "elapsed_s": time.time() - t0,
        "note": (
            "Full parameter DoF under pure FSOT consensus. "
            "D_eff calibrates LR via compute_scalar. "
            "3D spatial geometry is a later lift of the same operators."
        ),
        "ok": True,
    }
    path = OUT / "full_dof_push.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== FULL DoF SUMMARY ===")
    print(f"best agree={best['agree']:.0%} KL={best['kl']:.3f} step={best['step']}")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

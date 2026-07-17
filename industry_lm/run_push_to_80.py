#!/usr/bin/env python3
"""
Push pure-FSOT all-layer model from ~50% toward 80% agreement.
Loads pure_fsot_full_best.pt when present; else LoRA-bake path.
Packed multi-prompt batches + cosine LR + long run.
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
from lora_utils import LoRALinear, inject_lora_into_fsot_attn  # noqa: E402
from train_corpus import PROBES, train_texts  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm"
CKPT = OUT / "checkpoints"
CKPT.mkdir(parents=True, exist_ok=True)

STEPS = 6000
EVAL_EVERY = 200
LR_MAX = 1.2e-4
LR_MIN = 1.5e-5
TARGET = 0.80
BATCH_PROMPTS = 6  # pack several short prefixes per step


def load(device):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    return tok, m


def merge_lora(lora: LoRALinear) -> torch.nn.Linear:
    base = lora.base
    with torch.no_grad():
        delta = (lora.lora_B @ lora.lora_A) * lora.scaling
        w = base.weight.data + delta
    lin = torch.nn.Linear(
        base.in_features, base.out_features, bias=base.bias is not None
    ).to(device=base.weight.device, dtype=base.weight.dtype)
    lin.weight.data.copy_(w)
    if base.bias is not None:
        lin.bias.data.copy_(base.bias.data)
    return lin


def bake_lora(model):
    for layer in model.model.layers:
        attn = layer.self_attn
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            m = getattr(attn, attr)
            if isinstance(m, LoRALinear):
                setattr(attn, attr, merge_lora(m))


@torch.no_grad()
def measure(tok, teacher, student, device):
    agree = kl = top = 0.0
    n = len(PROBES)
    details = []
    for p in PROBES:
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


def pack_batch(tok, texts, device):
    """Pad a list of short prefixes into one batch."""
    enc = tok(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64,
    )
    return {k: v.to(device) for k, v in enc.items()}


def lr_at(step, total):
    # cosine decay
    t = step / max(total - 1, 1)
    return LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1 + math.cos(math.pi * t))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = train_texts()
    print(
        f"PUSH TO 80%  device={device} steps={STEPS} corpus={len(corpus)} "
        f"batch_prompts={BATCH_PROMPTS}"
    )

    tok, teacher = load(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    tps_b = tps(tok, teacher, device)
    print(f"baseline tps={tps_b:.1f}")

    _, student = load(device)
    swap_all_layers(student)

    push_ckpt = CKPT / "pure_fsot_push80_best.pt"
    full_ckpt = CKPT / "pure_fsot_full_best.pt"
    lora_ckpt = CKPT / "pure_fsot_lora_best.pt"
    if push_ckpt.is_file():
        ckpt = torch.load(push_ckpt, map_location=device, weights_only=False)
        missing, unexpected = student.load_state_dict(ckpt["state_dict"], strict=False)
        print(
            f"loaded push80 best step={ckpt.get('step')} agree={ckpt.get('agree')} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    elif full_ckpt.is_file():
        # rebuild full-weight student: swap then load trainable keys
        ckpt = torch.load(full_ckpt, map_location=device, weights_only=False)
        # full best may be after bake — load what we can
        missing, unexpected = student.load_state_dict(ckpt["state_dict"], strict=False)
        print(
            f"loaded full best step={ckpt.get('step')} agree={ckpt.get('agree')} "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    elif lora_ckpt.is_file():
        inject_lora_into_fsot_attn(student, r=16, alpha=32)
        student.to(device)
        ckpt = torch.load(lora_ckpt, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["state_dict"], strict=False)
        bake_lora(student)
        print(f"loaded+baked lora best agree={ckpt.get('agree')}")
    else:
        print("no checkpoint — cold pure FSOT")

    for n, p in student.named_parameters():
        train = any(
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
        p.requires_grad_(train)
    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"trainable {n_train:,}")

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=LR_MAX,
        weight_decay=0.01,
    )

    best = {"agree": -1.0, "kl": 1e9, "step": -1}
    history = []
    student.train()
    t0 = time.perf_counter()
    for step in range(STEPS):
        # packed batch of short prompts
        texts = [corpus[(step * BATCH_PROMPTS + i) % len(corpus)] for i in range(BATCH_PROMPTS)]
        inp = pack_batch(tok, texts, device)
        lr = lr_at(step, STEPS)
        for g in opt.param_groups:
            g["lr"] = lr

        with torch.no_grad():
            t_log = teacher(**inp).logits
        s_log = student(**inp).logits
        # mask pad positions
        mask = inp["attention_mask"].unsqueeze(-1).float()
        log_s = F.log_softmax(s_log, dim=-1)
        p_t = F.softmax(t_log, dim=-1)
        # token-mean KL
        kl_tok = (p_t * (torch.log(p_t.clamp_min(1e-12)) - log_s)).sum(dim=-1)
        loss = (kl_tok * mask.squeeze(-1)).sum() / mask.squeeze(-1).sum().clamp_min(1.0)

        if not torch.isfinite(loss):
            print(f"non-finite @ {step}")
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 1.0
        )
        opt.step()

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            student.eval()
            a, k, t5, details = measure(tok, teacher, student, device)
            tp = tps(tok, student, device)
            student.train()
            row = {
                "step": step,
                "loss": float(loss.item()),
                "lr": lr,
                "agree": a,
                "kl": k,
                "top5": t5,
                "tps_x": tp / max(tps_b, 1e-9),
            }
            history.append(row)
            print(
                f"  {step:04d} loss={loss.item():.3f} lr={lr:.2e} agree={a:.0%} "
                f"KL={k:.3f} top5={t5:.2f} tps×{tp/max(tps_b,1e-9):.2f}"
            )
            if a > best["agree"] or (abs(a - best["agree"]) < 1e-9 and k < best["kl"]):
                best = {
                    "agree": a,
                    "kl": k,
                    "step": step,
                    "top5": t5,
                    "tps": tp,
                    "details": details,
                }
                torch.save(
                    {
                        "step": step,
                        "agree": a,
                        "kl": k,
                        "top5": t5,
                        "state_dict": {
                            n: p.detach().cpu()
                            for n, p in student.named_parameters()
                            if p.requires_grad
                        },
                    },
                    CKPT / "pure_fsot_push80_best.pt",
                )
                print("    * BEST saved pure_fsot_push80_best.pt")
            if a >= TARGET:
                print(f"*** TARGET {TARGET:.0%} REACHED at step {step} ***")
                break

    elapsed = time.perf_counter() - t0
    student.eval()
    a, k, t5, details = measure(tok, teacher, student, device)
    tp = tps(tok, student, device)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "push_to_80_packed",
        "steps": STEPS,
        "batch_prompts": BATCH_PROMPTS,
        "elapsed_sec": elapsed,
        "baseline_tps": tps_b,
        "history": history,
        "best": {
            "agree": best["agree"],
            "kl": best["kl"],
            "step": best["step"],
            "top5": best.get("top5"),
            "tps": best.get("tps"),
            "details": best.get("details"),
        },
        "final": {
            "agree": a,
            "kl": k,
            "top5": t5,
            "tps": tp,
            "tps_x": tp / max(tps_b, 1e-9),
            "details": details,
        },
        "hit_80": best["agree"] >= TARGET,
        "climb": "0% → 25% → 50% → best now",
        "ok": True,
    }
    path = OUT / "push_to_80.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== PUSH 80 SUMMARY ===")
    print(
        f"best agree={best['agree']:.0%} KL={best['kl']:.3f} @ {best['step']} | "
        f"final={a:.0%} tps×{tp/max(tps_b,1e-9):.2f} HIT80={best['agree']>=TARGET}"
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

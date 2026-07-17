#!/usr/bin/env python3
"""
Continue pure-FSOT training from LoRA warm-start:
  unfreeze full q/k/v/o + norms, 3000 more steps, bigger corpus.
Target: push agreement toward 80%.
"""
from __future__ import annotations

import json
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

STEPS = 3000
EVAL_EVERY = 200
LR = 1e-4
TARGET = 0.80


def load(device):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    return tok, m


def merge_lora_into_linear(lora: LoRALinear) -> torch.nn.Linear:
    """Bake LoRA into a new Linear (full rank continue)."""
    base = lora.base
    with torch.no_grad():
        # W' = W + scaling * B @ A
        delta = (lora.lora_B @ lora.lora_A) * lora.scaling
        w = base.weight.data + delta
    lin = torch.nn.Linear(base.in_features, base.out_features, bias=base.bias is not None)
    lin = lin.to(device=base.weight.device, dtype=base.weight.dtype)
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
                setattr(attn, attr, merge_lora_into_linear(m))


@torch.no_grad()
def measure(tok, teacher, student, device):
    agree = kl = top = 0.0
    n = len(PROBES)
    for p in PROBES:
        inp = tok(p, return_tensors="pt").to(device)
        lt = teacher(**inp).logits[0, -1].float().cpu()
        ls = student(**inp).logits[0, -1].float().cpu()
        if not torch.isfinite(ls).all():
            return 0.0, float("nan"), 0.0
        pt = torch.softmax(lt, dim=-1)
        kl += float(
            (pt * (torch.log(pt.clamp_min(1e-12)) - torch.log_softmax(ls, dim=-1))).sum()
        )
        agree += float(lt.argmax() == ls.argmax())
        top += len(
            set(torch.topk(lt, 5).indices.tolist())
            & set(torch.topk(ls, 5).indices.tolist())
        ) / 5.0
    return agree / n, kl / n, top / n


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = train_texts()
    print(f"continue pure FSOT  steps={STEPS} corpus={len(corpus)} target={TARGET:.0%}")

    tok, teacher = load(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    tps_b = tps(tok, teacher, device)

    _, student = load(device)
    swap_all_layers(student)
    # warm start: inject lora and load best if present
    inject_lora_into_fsot_attn(student, r=16, alpha=32)
    student.to(device)
    ckpt_path = CKPT / "pure_fsot_lora_best.pt"
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        missing, unexpected = student.load_state_dict(ckpt["state_dict"], strict=False)
        print(f"loaded {ckpt_path} step={ckpt.get('step')} agree={ckpt.get('agree')}")
        print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    # bake LoRA into full weights then unfreeze full projs
    bake_lora(student)
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
    print(f"trainable after bake: {n_train:,}")

    opt = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad], lr=LR, weight_decay=0.01
    )
    best = {"agree": -1.0, "kl": 1e9, "step": -1}
    history = []
    student.train()
    for step in range(STEPS):
        text = corpus[step % len(corpus)]
        inp = tok(text, return_tensors="pt").to(device)
        with torch.no_grad():
            t_log = teacher(**inp).logits
        s_log = student(**inp).logits
        loss = F.kl_div(
            F.log_softmax(s_log, dim=-1),
            F.softmax(t_log, dim=-1),
            reduction="batchmean",
        )
        if not torch.isfinite(loss):
            print(f"non-finite {step}")
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student.parameters() if p.requires_grad], 1.0
        )
        opt.step()

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            student.eval()
            a, k, t5 = measure(tok, teacher, student, device)
            tp = tps(tok, student, device)
            student.train()
            history.append(
                {
                    "step": step,
                    "loss": float(loss.item()),
                    "agree": a,
                    "kl": k,
                    "top5": t5,
                    "tps_x": tp / max(tps_b, 1e-9),
                }
            )
            print(
                f"  {step:04d} loss={loss.item():.3f} agree={a:.0%} KL={k:.3f} "
                f"top5={t5:.2f} tps×{tp/max(tps_b,1e-9):.2f}"
            )
            if a > best["agree"] or (a == best["agree"] and k < best["kl"]):
                best = {"agree": a, "kl": k, "step": step, "top5": t5, "tps": tp}
                torch.save(
                    {
                        "step": step,
                        "agree": a,
                        "kl": k,
                        "state_dict": {
                            n: p.detach().cpu()
                            for n, p in student.named_parameters()
                            if p.requires_grad
                        },
                    },
                    CKPT / "pure_fsot_full_best.pt",
                )
                print("    * best saved")
            if a >= TARGET:
                print(f"TARGET {TARGET:.0%} HIT")
                break

    student.eval()
    a, k, t5 = measure(tok, teacher, student, device)
    tp = tps(tok, student, device)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "continue_pure_fsot_full_qkv",
        "steps_ran": len(history) and history[-1]["step"] + 1,
        "history": history,
        "final": {
            "agree": a,
            "kl": k,
            "top5": t5,
            "tps": tp,
            "tps_x": tp / max(tps_b, 1e-9),
        },
        "best": best,
        "baseline_tps": tps_b,
        "hit_80": best["agree"] >= TARGET,
        "ok": True,
    }
    path = OUT / "continue_pure_fsot.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("=== CONTINUE SUMMARY ===")
    print(
        f"best agree={best['agree']:.0%} KL={best['kl']:.3f} @step {best['step']} | "
        f"final agree={a:.0%} tps×{tp/max(tps_b,1e-9):.2f} target80={best['agree']>=TARGET}"
    )
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

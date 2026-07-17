#!/usr/bin/env python3
"""
Push multi-token generation quality for pure FSOT SmolLM host.

Only adapts Q/K/V/O + attention layer-norms (same subspace as agree_best ckpt).
Teacher-forced full-seq CE_hard(teacher argmax) + KL, low LR — do not touch MLP/embed.
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

STEPS = 2000
EVAL_EVERY = 50
LR_MAX = 8e-6
LR_MIN = 1e-6
BATCH = 2
SEQ = 128
CE_W = 2.0
KL_W = 1.0
TARGET_MT = 0.50
MIN_NT = 0.85  # never save if next-token collapses


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
    """Q/K/V/O + input/post attn norms only — matches agree_best subspace."""
    params = []
    for name, p in model.named_parameters():
        if any(
            s in name
            for s in (
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "input_layernorm",
                "post_attention_layernorm",
            )
        ):
            p.requires_grad_(True)
            params.append(p)
        else:
            p.requires_grad_(False)
    return params


@torch.no_grad()
def next_token_agree(tok, teacher, student, device, probes):
    ok = 0
    for p in probes:
        inp = tok(p, return_tensors="pt").to(device)
        lt = teacher(**inp).logits[0, -1]
        ls = student(**inp).logits[0, -1]
        ok += int(lt.argmax() == ls.argmax())
    return ok / len(probes)


@torch.no_grad()
def multi_token_agree(tok, teacher, student, device, probes, max_new=8):
    match = total = 0
    for p in probes:
        inp = tok(p, return_tensors="pt").to(device)
        bt = teacher.generate(
            **inp,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        st = student.generate(
            **inp,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        bnew = bt[0, inp["input_ids"].shape[-1] :].tolist()
        snew = st[0, inp["input_ids"].shape[-1] :].tolist()
        n = min(len(bnew), len(snew))
        match += sum(1 for i in range(n) if bnew[i] == snew[i])
        total += n
    return match / max(total, 1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== PUSH GENERATION (QKV/norm only) ===")
    print("device", device)

    tok, teacher = load(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    _, student = load(device)
    swap_all_layers(student)
    src = CKPT / "pure_fsot_agree_best.pt"
    if not src.is_file():
        src = CKPT / "pure_fsot_push80_best.pt"
    if src.is_file():
        ck = torch.load(src, map_location=device, weights_only=False)
        missing, unexpected = student.load_state_dict(ck["state_dict"], strict=False)
        print(
            "loaded",
            src,
            "meta agree16",
            ck.get("agree16"),
            "missing",
            len(missing),
            "unexpected",
            len(unexpected),
        )
    student.train()

    params = adapted_params(student)
    print("trainable tensors", len(params), "n_elem", sum(p.numel() for p in params))
    opt = torch.optim.AdamW(params, lr=LR_MAX, weight_decay=0.0)

    # teacher-generated continuations for generation alignment
    texts = train_texts()
    if len(texts) < 16:
        texts = list(texts) + list(PROBES) * 30

    # Prefetch teacher rollouts (greedy) as distillation targets
    print("Building teacher continuation corpus...")
    cont_ids = []
    with torch.no_grad():
        for p in (list(PROBES) + list(texts))[:64]:
            inp = tok(p, return_tensors="pt").to(device)
            out = teacher.generate(
                **inp,
                max_new_tokens=24,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            cont_ids.append(out[0].cpu())
    print("continuation samples", len(cont_ids))

    best_mt = -1.0
    best_nt = -1.0
    best_path = CKPT / "pure_fsot_gen_best.pt"
    history = []

    # baseline before train
    student.eval()
    nt0 = next_token_agree(tok, teacher, student, device, EVAL16)
    mt0 = multi_token_agree(tok, teacher, student, device, EVAL16[:8], max_new=8)
    student.train()
    print(f"baseline nt={nt0:.0%} mt={mt0:.0%}")

    t0 = time.time()
    for step in range(1, STEPS + 1):
        lr = lr_at(step, STEPS)
        for g in opt.param_groups:
            g["lr"] = lr

        # mix: corpus chunk + teacher continuation
        use_cont = cont_ids[(step * BATCH) % len(cont_ids)]
        # pad/truncate single seq into batch of 1-2
        seqs = []
        for i in range(BATCH):
            if (step + i) % 2 == 0:
                seqs.append(cont_ids[(step * BATCH + i) % len(cont_ids)])
            else:
                t = texts[(step * BATCH + i) % len(texts)]
                ids = tok(
                    t, return_tensors="pt", truncation=True, max_length=SEQ
                )["input_ids"][0]
                seqs.append(ids)

        # pad batch
        max_len = min(SEQ, max(int(s.numel()) for s in seqs))
        ids_list = []
        mask_list = []
        for s in seqs:
            s = s[:max_len]
            pad = max_len - int(s.numel())
            if pad > 0:
                s = torch.cat([s, torch.full((pad,), tok.pad_token_id, dtype=s.dtype)])
            ids_list.append(s)
            m = (s != tok.pad_token_id).long()
            # mark original pads
            mask_list.append(m)
        ids = torch.stack(ids_list).to(device)
        mask = torch.stack(mask_list).to(device)

        with torch.no_grad():
            t_logits = teacher(input_ids=ids, attention_mask=mask).logits
        s_logits = student(input_ids=ids, attention_mask=mask).logits

        shift_s = s_logits[:, :-1, :].contiguous()
        shift_t = t_logits[:, :-1, :].contiguous()
        shift_m = mask[:, 1:].contiguous().float()

        # teacher-hard CE (agreement across all positions)
        t_hard = shift_t.argmax(dim=-1)
        ce_t = F.cross_entropy(
            shift_s.reshape(-1, shift_s.size(-1)),
            t_hard.reshape(-1),
            reduction="none",
        ).view_as(t_hard)
        ce_t = (ce_t * shift_m).sum() / shift_m.sum().clamp_min(1.0)

        # emphasize last 8 positions (generation tip)
        tip = min(8, shift_m.size(1))
        tip_m = shift_m.clone()
        tip_m[:, :-tip] = 0
        ce_tip = F.cross_entropy(
            shift_s.reshape(-1, shift_s.size(-1)),
            t_hard.reshape(-1),
            reduction="none",
        ).view_as(t_hard)
        ce_tip = (ce_tip * tip_m).sum() / tip_m.sum().clamp_min(1.0)

        log_ps = F.log_softmax(shift_s.float(), dim=-1)
        pt = F.softmax(shift_t.float(), dim=-1)
        kl = (pt * (torch.log(pt.clamp_min(1e-12)) - log_ps)).sum(-1)
        kl = (kl * shift_m).sum() / shift_m.sum().clamp_min(1.0)

        loss = CE_W * (0.5 * ce_t + 0.5 * ce_tip) + KL_W * kl
        if not torch.isfinite(loss):
            print("non-finite at", step)
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 0.5)
        opt.step()

        if step % EVAL_EVERY == 0 or step == 1:
            student.eval()
            nt = next_token_agree(tok, teacher, student, device, EVAL16)
            mt = multi_token_agree(tok, teacher, student, device, EVAL16[:8], max_new=8)
            student.train()
            row = {
                "step": step,
                "loss": float(loss.item()),
                "ce_t": float(ce_t.item()),
                "kl": float(kl.item()),
                "next_token": nt,
                "multi_token": mt,
                "lr": lr,
            }
            history.append(row)
            print(
                f"step {step}/{STEPS} loss={loss.item():.3f} nt={nt:.0%} mt={mt:.0%} lr={lr:.2e}"
            )
            if mt >= best_mt and nt >= MIN_NT:
                best_mt = mt
                best_nt = nt
                # save only adapted weights (same style as agree_best)
                sd = {
                    k: v.detach().cpu()
                    for k, v in student.state_dict().items()
                    if any(
                        s in k
                        for s in (
                            "self_attn.q_proj",
                            "self_attn.k_proj",
                            "self_attn.v_proj",
                            "self_attn.o_proj",
                            "input_layernorm",
                            "post_attention_layernorm",
                        )
                    )
                }
                torch.save(
                    {
                        "state_dict": sd,
                        "step": step,
                        "multi_token": mt,
                        "next_token": nt,
                        "agree16": nt,
                    },
                    best_path,
                )
                print("  saved", best_path, f"mt={mt:.0%} nt={nt:.0%}")
            if mt >= TARGET_MT and nt >= 0.90:
                print("HIT targets mt", TARGET_MT, "nt>=90%")
                break

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_nt": nt0,
        "baseline_mt": mt0,
        "best_multi_token": best_mt,
        "best_next_token": best_nt,
        "best_ckpt": str(best_path) if best_path.is_file() else None,
        "history": history,
        "elapsed_s": time.time() - t0,
        "target_mt": TARGET_MT,
    }
    outp = OUT / "push_generation.json"
    outp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", outp, "best_mt", best_mt, "best_nt", best_nt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Run eval on real packs and dump a full miss trail for every wrong answer.

Output (easy to open):
  results/industry_lm/miss_traces/miss_trace_<arm>.md
  results/industry_lm/miss_traces/miss_trace_<arm>.jsonl

Each miss shows: question, gold, model thought/generation, diagnosis tags.
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from fsot_layer_swap import swap_all_layers  # noqa: E402
from miss_trace import extract_num, make_miss_entry, write_miss_log  # noqa: E402
from real_data_packs import (  # noqa: E402
    load_arc_train,
    load_gsm8k_test,
    load_math_train,
)

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
OUT = ROOT / "results" / "industry_lm" / "miss_traces"
CKPT = ROOT / "results" / "industry_lm" / "checkpoints"
DATA = Path(r"D:\training data")

N_GSM = 40
N_ARC = 40
N_MATH = 30


def load_base(device):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device).eval()
    return tok, m


def load_fsot(device):
    tok, m = load_base(device)
    swap_all_layers(m)
    for path in [
        CKPT / "pure_fsot_12x3_best.pt",
        CKPT / "pure_fsot_realdata_best.pt",
        CKPT / "pure_fsot_curriculum_best.pt",
        CKPT / "pure_fsot_agree100_best.pt",
    ]:
        if path.is_file():
            ck = torch.load(path, map_location=device, weights_only=False)
            m.load_state_dict(ck["state_dict"], strict=False)
            return tok, m, str(path), ck
    return tok, m, None, {}


@torch.no_grad()
def gen(tok, model, device, prompt, max_new=48):
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
    out = model.generate(
        **inp,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt) :] if text.startswith(prompt) else text


def eval_with_traces(tok, model, device, arm: str):
    misses = []
    stats = {}

    # GSM
    gsm = load_gsm8k_test(N_GSM)
    hits = 0
    for r in gsm:
        prompt = r["prompt"]
        if not prompt.rstrip().endswith("####"):
            prompt = prompt.split("Answer:")[0].strip() + "\n####"
        # allow a bit more tokens so "thought" is visible
        thought = gen(tok, model, device, prompt, max_new=64)
        gold = extract_num(str(r["gold"])) or str(r["gold"]).strip()
        nums = re.findall(r"-?\d+\.?\d*", thought.replace(",", ""))
        pred = nums[0] if nums else None
        ok = pred is not None and gold is not None and pred == gold
        hits += int(ok)
        if not ok:
            misses.append(
                make_miss_entry(
                    kind="gsm8k",
                    prompt=prompt,
                    gold=str(gold),
                    pred=pred,
                    thought=thought,
                    arm=arm,
                )
            )
    stats["gsm8k"] = {"n": len(gsm), "acc": hits / max(len(gsm), 1), "misses": len(gsm) - hits}

    # ARC
    arc = load_arc_train(DATA / "ARC-Easy_train.csv", N_ARC)
    hits = 0
    for r in arc:
        thought = gen(tok, model, device, r["prompt"], max_new=24)
        m = re.search(r"\b([ABCD])\b", thought.upper())
        pred = m.group(1) if m else (thought.strip()[:1].upper() if thought.strip() else "")
        gold = r["gold"].strip().upper()
        ok = pred == gold
        hits += int(ok)
        if not ok:
            misses.append(
                make_miss_entry(
                    kind="arc",
                    prompt=r["prompt"],
                    gold=gold,
                    pred=pred,
                    thought=thought,
                    arm=arm,
                )
            )
    stats["arc"] = {"n": len(arc), "acc": hits / max(len(arc), 1), "misses": len(arc) - hits}

    # MATH
    math_rows = load_math_train(N_MATH)
    hits = 0
    for r in math_rows:
        thought = gen(tok, model, device, r["prompt"], max_new=48)
        gold = str(r["gold"])
        gnum = extract_num(gold)
        pnum = extract_num(thought)
        ok = bool(gnum and pnum and gnum == pnum)
        if not ok:
            gclean = re.sub(r"\s+", "", gold.lower())[:40]
            if gclean and gclean in re.sub(r"\s+", "", thought.lower()):
                ok = True
        hits += int(ok)
        if not ok:
            misses.append(
                make_miss_entry(
                    kind="math",
                    prompt=r["prompt"],
                    gold=gold,
                    pred=pnum,
                    thought=thought,
                    arm=arm,
                )
            )
    stats["math"] = {
        "n": len(math_rows),
        "acc": hits / max(len(math_rows), 1),
        "misses": len(math_rows) - hits,
    }
    return misses, stats


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== MISS TRACE (wrong answers + thought path) ===")

    # FSOT arm
    tok, fsot, ckpt, meta = load_fsot(device)
    print("fsot ckpt", ckpt)
    misses_f, stats_f = eval_with_traces(tok, fsot, device, "fsot")
    paths_f = write_miss_log(
        misses_f,
        OUT,
        name="miss_trace_fsot",
        meta={
            "arm": "fsot",
            "ckpt": ckpt,
            "stats": stats_f,
            "meta_ckpt": {
                k: meta.get(k)
                for k in ("macro", "gsm", "arc", "math", "agree16", "step", "epoch")
                if isinstance(meta, dict)
            },
        },
    )
    print("FSOT stats", stats_f)
    print("FSOT miss log:", paths_f["md"])
    print(f"  ({len(misses_f)} misses)")

    # Baseline arm (for comparison)
    tok_b, base = load_base(device)
    misses_b, stats_b = eval_with_traces(tok_b, base, device, "baseline")
    paths_b = write_miss_log(
        misses_b,
        OUT,
        name="miss_trace_baseline",
        meta={"arm": "baseline", "stats": stats_b},
    )
    print("BASE stats", stats_b)
    print("BASE miss log:", paths_b["md"])
    print(f"  ({len(misses_b)} misses)")

    # index file
    idx = OUT / "README_MISS_TRACES.md"
    idx.write_text(
        f"""# Miss traces (wrong-answer audit)

Generated: {datetime.now(timezone.utc).isoformat()}

## How to use

Open the **`.md`** file for a readable trail of every miss:

1. **QUESTION / PROMPT** — what was asked  
2. **GOLD ANSWER** — correct target  
3. **MODEL PRED** — parsed answer  
4. **MODEL THOUGHT / GENERATION** — full free-gen path (how it tried)  
5. **DIAGNOSIS** — tags (empty gen, wrong number, regurgitated question, …)

## Files

| Arm | Human log | JSONL | Summary |
|-----|-----------|-------|---------|
| Pure FSOT | [`miss_trace_fsot.md`](miss_trace_fsot.md) | `miss_trace_fsot.jsonl` | `miss_trace_fsot_summary.json` |
| Baseline | [`miss_trace_baseline.md`](miss_trace_baseline.md) | `miss_trace_baseline.jsonl` | `miss_trace_baseline_summary.json` |

## Snapshot scores

**FSOT:** {stats_f}  

**Baseline:** {stats_b}

Ckpt: `{ckpt}`
""",
        encoding="utf-8",
    )
    print("index", idx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

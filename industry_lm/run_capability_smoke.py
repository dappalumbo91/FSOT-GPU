#!/usr/bin/env python3
"""
Public capability smoke: ARC-Easy, GSM8K, MATH subsets vs industry baseline.

Same pure-FSOT host vs SDPA baseline on this GPU. Honest ledgers for open-source SOTA.
"""
from __future__ import annotations

import ast
import csv
import json
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

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
DATA = Path(r"D:\training data")
OUT = ROOT / "results" / "industry_lm"
CKPT = OUT / "checkpoints"
OUT.mkdir(parents=True, exist_ok=True)

N_ARC = 40
N_GSM = 40
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
        CKPT / "pure_fsot_sota_climb_best.pt",
        CKPT / "pure_fsot_curriculum_best.pt",
        CKPT / "pure_fsot_exceed_best.pt",
        CKPT / "pure_fsot_agree100_best.pt",
    ]:
        if path.is_file():
            ck = torch.load(path, map_location=device, weights_only=False)
            m.load_state_dict(ck["state_dict"], strict=False)
            return tok, m, str(path), ck
    return tok, m, None, {}


@torch.no_grad()
def gen(tok, model, device, prompt, max_new=32):
    inp = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inp,
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    text = tok.decode(out[0], skip_special_tokens=True)
    if text.startswith(prompt):
        return text[len(prompt) :]
    return text


def extract_num(s: str):
    # last number in string
    nums = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
    return nums[-1] if nums else None


def load_arc(n=N_ARC):
    path = DATA / "ARC-Easy_train.csv"
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i >= n:
                break
            # choices stored as python-ish dict string — fragile; parse carefully
            q = row["question"]
            key = row["answerKey"].strip()
            ch = row["choices"]
            labels, texts = [], []
            try:
                # try eval-like for the array form
                m_lab = re.search(r"'label':\s*array\(\[(.*?)\]", ch, re.S)
                m_txt = re.search(r"'text':\s*array\(\[(.*?)\]", ch, re.S)
                if m_lab and m_txt:
                    labels = re.findall(r"'([A-D])'", m_lab.group(1))
                    texts = re.findall(r"'([^']*)'", m_txt.group(1))
                else:
                    labels = ["A", "B", "C", "D"]
                    texts = [ch[:80]] * 4
            except Exception:
                labels = ["A", "B", "C", "D"]
                texts = ["?"] * 4
            opts = "\n".join(f"{lab}. {tx}" for lab, tx in zip(labels, texts))
            prompt = f"Question: {q}\n{opts}\nAnswer:"
            rows.append({"prompt": prompt, "answer": key, "kind": "arc_easy"})
    return rows


def load_gsm(n=N_GSM):
    path = DATA / "gsm8k" / "test.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            o = json.loads(line)
            # answer often ends with #### number
            ans = o["answer"]
            gold = ans.split("####")[-1].strip()
            prompt = f"Question: {o['question']}\nAnswer:"
            rows.append({"prompt": prompt, "answer": gold, "kind": "gsm8k"})
    return rows


def load_math(n=N_MATH):
    path = DATA / "math" / "math.jsonl"
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            # flexible keys
            q = o.get("problem") or o.get("question") or o.get("prompt") or ""
            a = o.get("solution") or o.get("answer") or o.get("target") or ""
            if not q:
                continue
            gold = extract_num(str(a)) or str(a)[:40]
            prompt = f"Problem: {q}\nFinal answer:"
            rows.append({"prompt": prompt, "answer": str(gold), "kind": "math"})
    return rows


@torch.no_grad()
def eval_set(tok, model, device, rows, max_new=48):
    hits = 0
    details = []
    for r in rows:
        tail = gen(tok, model, device, r["prompt"], max_new=max_new)
        gold = str(r["answer"]).strip()
        ok = False
        if r["kind"] == "arc_easy":
            # first letter A-D in gen
            m = re.search(r"\b([ABCD])\b", tail.upper())
            pred = m.group(1) if m else tail.strip()[:1].upper()
            ok = pred == gold.upper()
        else:
            pred = extract_num(tail) or tail.strip()[:20]
            gnum = extract_num(gold) or gold
            ok = pred is not None and gnum is not None and (
                pred == gnum or pred in gold or gnum in tail
            )
        hits += int(ok)
        details.append(
            {
                "kind": r["kind"],
                "gold": gold,
                "pred": (pred if r["kind"] == "arc_easy" else (extract_num(tail) or tail[:40])),
                "hit": ok,
            }
        )
    return hits / max(len(rows), 1), details


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== CAPABILITY SMOKE (open-source SOTA packs) ===")
    tok, base = load_base(device)
    tok2, fsot, ckpt, meta = load_fsot(device)
    print("ckpt", ckpt)

    arc = load_arc()
    gsm = load_gsm()
    math_rows = load_math()
    print(f"n_arc={len(arc)} n_gsm={len(gsm)} n_math={len(math_rows)}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ckpt": ckpt,
        "meta": {
            k: meta.get(k)
            for k in ("agree16", "fact_rate", "fsot_literacy", "step", "score")
            if isinstance(meta, dict)
        },
        "sets": {},
        "ok": True,
    }

    for name, rows in [("arc_easy", arc), ("gsm8k", gsm), ("math", math_rows)]:
        if not rows:
            report["sets"][name] = {"n": 0, "note": "empty"}
            continue
        print(f"eval {name} baseline...")
        b, _ = eval_set(tok, base, device, rows)
        print(f"eval {name} fsot...")
        f, det = eval_set(tok2, fsot, device, rows)
        report["sets"][name] = {
            "n": len(rows),
            "baseline": b,
            "fsot": f,
            "delta": f - b,
            "fsot_wins": f > b + 1e-9,
            "sample_misses": [d for d in det if not d["hit"]][:5],
        }
        print(f"  {name}: base={b:.0%} fsot={f:.0%} delta={f-b:+.0%}")

    # aggregate
    scores_b = [v["baseline"] for v in report["sets"].values() if "baseline" in v]
    scores_f = [v["fsot"] for v in report["sets"].values() if "fsot" in v]
    report["macro_baseline"] = sum(scores_b) / max(len(scores_b), 1)
    report["macro_fsot"] = sum(scores_f) / max(len(scores_f), 1)
    report["macro_delta"] = report["macro_fsot"] - report["macro_baseline"]
    report["fsot_macro_wins"] = report["macro_fsot"] > report["macro_baseline"]

    path = OUT / "capability_smoke.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = OUT / "CAPABILITY_SMOKE.md"
    lines = [
        "# Capability smoke — pure FSOT vs industry baseline (SmolLM2-135M)",
        "",
        f"Ckpt: `{ckpt}`",
        "",
        "| Set | n | Baseline | Pure FSOT | Δ |",
        "|-----|---|----------|-----------|---|",
    ]
    for k, v in report["sets"].items():
        if "baseline" not in v:
            continue
        lines.append(
            f"| {k} | {v['n']} | {v['baseline']:.0%} | {v['fsot']:.0%} | {v['delta']:+.0%} |"
        )
    lines.append("")
    lines.append(
        f"**Macro:** base {report['macro_baseline']:.0%} | FSOT **{report['macro_fsot']:.0%}** | "
        f"Δ {report['macro_delta']:+.0%} | win={report['fsot_macro_wins']}"
    )
    md.write_text("\n".join(lines), encoding="utf-8")
    print("macro", report["macro_baseline"], "->", report["macro_fsot"])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

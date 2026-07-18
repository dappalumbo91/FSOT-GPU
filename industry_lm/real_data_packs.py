#!/usr/bin/env python3
"""Load real training packs from D:\\training data (not synthetic probes)."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

DATA = Path(r"D:\training data")


def _parse_arc_choices(ch: str) -> tuple[list[str], list[str]]:
    labels = re.findall(r"'([A-D])'", ch)
    # texts inside array([...])
    m = re.search(r"'text':\s*array\(\[(.*?)\]\s*,\s*dtype", ch, re.S)
    if not m:
        m = re.search(r"'text':\s*array\(\[(.*?)\]", ch, re.S)
    texts = re.findall(r"'([^']*)'", m.group(1)) if m else []
    if len(labels) < 4:
        labels = ["A", "B", "C", "D"]
    while len(texts) < len(labels):
        texts.append("?")
    return labels[:4], texts[:4]


def load_gsm8k_train(limit: int | None = None) -> list[dict]:
    path = DATA / "gsm8k" / "train.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            o = json.loads(line)
            final = o["answer"].split("####")[-1].strip()
            # Compact real GSM format: full chain + forced final marker.
            # Also a short variant so small models learn #### early.
            text_full = (
                f"Question: {o['question']}\n"
                f"Answer: {o['answer'].strip()}\n"
                f"#### {final}"
            )
            text_short = f"Question: {o['question']}\n#### {final}"
            rows.append(
                {
                    "kind": "gsm8k",
                    "text": text_full,
                    "prompt": f"Question: {o['question']}\n####",
                    "gold": final,
                }
            )
            rows.append(
                {
                    "kind": "gsm8k",
                    "text": text_short,
                    "prompt": f"Question: {o['question']}\n####",
                    "gold": final,
                }
            )
    return rows


def load_gsm8k_test(limit: int = 40) -> list[dict]:
    path = DATA / "gsm8k" / "test.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            o = json.loads(line)
            final = o["answer"].split("####")[-1].strip()
            rows.append(
                {
                    "kind": "gsm8k",
                    "prompt": f"Question: {o['question']}\n####",
                    "gold": final,
                    "text": f"Question: {o['question']}\n#### {final}",
                }
            )
    return rows


def load_arc_train(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if limit is not None and i >= limit:
                break
            labels, texts = _parse_arc_choices(row["choices"])
            opts = "\n".join(f"{lab}. {tx}" for lab, tx in zip(labels, texts))
            key = row["answerKey"].strip()
            # find answer text
            ans_txt = key
            for lab, tx in zip(labels, texts):
                if lab == key:
                    ans_txt = f"{lab}. {tx}"
                    break
            q = row["question"]
            prompt = f"Question: {q}\n{opts}\nAnswer:"
            text = f"{prompt} {ans_txt}"
            rows.append(
                {
                    "kind": "arc",
                    "prompt": prompt,
                    "gold": key,
                    "text": text,
                }
            )
    return rows


def load_math_train(limit: int | None = None) -> list[dict]:
    path = DATA / "math" / "math.jsonl"
    rows = []
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            prob = o.get("problem") or ""
            ans = str(o.get("answer") or "")
            sol = str(o.get("solution") or "")[:800]
            if not prob:
                continue
            prompt = f"Problem: {prob}\nFinal answer:"
            text = f"{prompt} {ans}\nSolution: {sol}"
            rows.append(
                {
                    "kind": "math",
                    "prompt": prompt,
                    "gold": ans,
                    "text": text,
                }
            )
    return rows


def build_train_mix(
    n_gsm: int = 4000,
    n_arc_easy: int = 3000,
    n_arc_hard: int = 1500,
    n_math: int = 500,
) -> list[dict]:
    rows = []
    rows.extend(load_gsm8k_train(n_gsm))
    rows.extend(load_arc_train(DATA / "ARC-Easy_train.csv", n_arc_easy))
    rows.extend(load_arc_train(DATA / "ARC-Challenge_train.csv", n_arc_hard))
    rows.extend(load_math_train(n_math))
    return rows

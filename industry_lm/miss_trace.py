#!/usr/bin/env python3
"""
Wrong-answer trace log: question, gold, model thought/generation, diagnosis.

When the host gets an item wrong we dump a human-readable trail so failures
are audit-able (not just a % score).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def extract_num(s: str) -> str | None:
    nums = re.findall(r"-?\d+\.?\d*", s.replace(",", ""))
    return nums[-1] if nums else None


def diagnose_failure(
    *,
    kind: str,
    question: str,
    gold: str,
    pred: str | None,
    thought: str,
) -> list[str]:
    """Heuristic tags for how the model went wrong (audit aid, not theory)."""
    tags: list[str] = []
    t = thought or ""
    tl = t.lower()
    ql = question.lower()

    if not t.strip():
        tags.append("empty_generation")
    if pred is None or str(pred).strip() == "":
        tags.append("no_parseable_answer")
    if kind in ("gsm8k", "math"):
        if "####" not in t and kind == "gsm8k":
            tags.append("missing_final_marker_####")
        # regurgitated question
        q_snip = re.sub(r"\s+", " ", question)[:40].lower()
        if q_snip and q_snip in tl:
            tags.append("question_regurgitation")
        # answer digits appear in question (classic leakage trap if scorer is loose)
        g = extract_num(str(gold))
        if g and g in re.findall(r"-?\d+\.?\d*", question.replace(",", "")):
            tags.append("gold_digit_also_in_question")
        if pred and g and pred != g:
            tags.append("wrong_numeric_final")
        if re.search(r"(answer is|final answer)", tl):
            tags.append("natural_language_answer_phrase")
    if kind in ("arc", "arc_easy"):
        if not re.search(r"\b[ABCD]\b", t.upper()):
            tags.append("no_letter_choice")
        if pred and gold and pred.upper() != str(gold).upper():
            tags.append("wrong_choice_letter")
    if len(t) < 3:
        tags.append("too_short_trace")
    if len(t) > 400:
        tags.append("long_rambling")
    if not tags:
        tags.append("unclassified_wrong")
    return tags


def format_miss_block(entry: dict[str, Any], index: int) -> str:
    tags = entry.get("diagnosis") or []
    thought = entry.get("thought") or entry.get("generation") or ""
    return "\n".join(
        [
            f"{'=' * 72}",
            f"MISS #{index}  kind={entry.get('kind')}  arm={entry.get('arm', 'fsot')}",
            f"{'=' * 72}",
            "",
            "QUESTION / PROMPT:",
            entry.get("question") or entry.get("prompt") or "",
            "",
            f"GOLD ANSWER:  {entry.get('gold')}",
            f"MODEL PRED:   {entry.get('pred')}",
            "",
            "MODEL THOUGHT / GENERATION (how it tried to solve it):",
            "-----",
            thought if thought.strip() else "(empty)",
            "-----",
            "",
            f"DIAGNOSIS: {', '.join(tags)}",
            "",
        ]
    )


def write_miss_log(
    misses: list[dict[str, Any]],
    out_dir: Path,
    *,
    name: str = "miss_trace",
    meta: dict[str, Any] | None = None,
) -> dict[str, str]:
    """
    Write:
      - {name}.md   human readable
      - {name}.jsonl machine trail (one miss per line)
      - {name}_summary.json counts by diagnosis
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    md_path = out_dir / f"{name}.md"
    jl_path = out_dir / f"{name}.jsonl"
    sum_path = out_dir / f"{name}_summary.json"

    # enrich diagnosis
    for m in misses:
        if "diagnosis" not in m:
            m["diagnosis"] = diagnose_failure(
                kind=str(m.get("kind") or ""),
                question=str(m.get("question") or m.get("prompt") or ""),
                gold=str(m.get("gold") or ""),
                pred=None if m.get("pred") is None else str(m.get("pred")),
                thought=str(m.get("thought") or m.get("generation") or ""),
            )
        m["timestamp"] = m.get("timestamp") or ts

    tag_counts: dict[str, int] = {}
    kind_counts: dict[str, int] = {}
    for m in misses:
        kind_counts[str(m.get("kind"))] = kind_counts.get(str(m.get("kind")), 0) + 1
        for t in m.get("diagnosis") or []:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    # markdown
    lines = [
        f"# Wrong-answer miss trace — {name}",
        "",
        f"**UTC:** {ts}",
        f"**Misses logged:** {len(misses)}",
        "",
    ]
    if meta:
        lines.append("## Meta")
        lines.append("```json")
        lines.append(json.dumps(meta, indent=2, default=str))
        lines.append("```")
        lines.append("")
    lines.append("## Diagnosis counts")
    lines.append("")
    for k, v in sorted(tag_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## By kind")
    lines.append("")
    for k, v in sorted(kind_counts.items()):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Misses (full trails)")
    lines.append("")
    for i, m in enumerate(misses, 1):
        lines.append(format_miss_block(m, i))

    md_path.write_text("\n".join(lines), encoding="utf-8")
    with jl_path.open("w", encoding="utf-8") as f:
        for m in misses:
            f.write(json.dumps(m, ensure_ascii=False, default=str) + "\n")
    summary = {
        "timestamp": ts,
        "n_misses": len(misses),
        "by_kind": kind_counts,
        "by_diagnosis": tag_counts,
        "meta": meta or {},
        "paths": {"md": str(md_path), "jsonl": str(jl_path)},
    }
    sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"md": str(md_path), "jsonl": str(jl_path), "summary": str(sum_path)}


def make_miss_entry(
    *,
    kind: str,
    prompt: str,
    gold: str,
    pred: str | None,
    thought: str,
    arm: str = "fsot",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "kind": kind,
        "arm": arm,
        "question": prompt,
        "prompt": prompt,
        "gold": gold,
        "pred": pred,
        "thought": thought,
        "generation": thought,
    }
    entry["diagnosis"] = diagnose_failure(
        kind=kind,
        question=prompt,
        gold=gold,
        pred=pred,
        thought=thought,
    )
    if extra:
        entry.update(extra)
    return entry

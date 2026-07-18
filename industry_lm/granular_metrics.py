#!/usr/bin/env python3
"""
Granular accuracy metrics beyond single GSM%/ARC% headlines.

Axes:
  GSM free-gen: exact, first-digit, len-match, format-ok, mode-collapse
  GSM TF: next-token gold (space-aligned and bare digit)
  GSM constrained decode: digits-only free-gen exact
  ARC free-gen: letter exact, first-token TF letter, Easy vs Challenge, held-out
  Bucketed: by gold digit-length, by letter
  Agree16: next-token fidelity vs teacher
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import torch


def extract_num(s: str) -> str | None:
    nums = re.findall(r"-?\d+\.?\d*", str(s).replace(",", ""))
    return nums[-1] if nums else None


def normalize_num(s: str | None) -> str | None:
    if s is None:
        return None
    s = str(s).strip().replace(",", "")
    if not s:
        return None
    try:
        if "." in s:
            f = float(s)
            if f == int(f):
                return str(int(f))
            return str(f)
        return str(int(s))
    except ValueError:
        return s


@torch.no_grad()
def free_gen(tok, model, device, prompt: str, max_new: int = 8, eos_ids=None) -> str:
    inp = tok(prompt, return_tensors="pt", truncation=True, max_length=480).to(device)
    kw = dict(
        max_new_tokens=max_new,
        do_sample=False,
        pad_token_id=tok.pad_token_id,
    )
    if eos_ids:
        kw["eos_token_id"] = eos_ids
    out = model.generate(**inp, **kw)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(prompt) :] if text.startswith(prompt) else text


def digit_token_ids(tok) -> list[int]:
    ids = []
    for d in "0123456789":
        e = tok.encode(d, add_special_tokens=False)
        if len(e) == 1:
            ids.append(e[0])
    space = tok.encode(" ", add_special_tokens=False)
    nl = tok.encode("\n", add_special_tokens=False)
    if space:
        ids.append(space[0])
    if nl:
        ids.append(nl[0])
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    return sorted(set(ids))


@torch.no_grad()
def constrained_digit_gen(
    tok, model, device, prompt: str, max_new: int = 8, allow_ids: list[int] | None = None
) -> str:
    """
    Greedy decode restricted to space/digits/newline/EOS.
    Measures 'can the model rank the right digits when format soup is blocked.'
    """
    allow = set(allow_ids or digit_token_ids(tok))
    pe = tok(prompt, return_tensors="pt", truncation=True, max_length=480).to(device)
    ids = pe["input_ids"]
    attn = pe["attention_mask"]
    gen_ids: list[int] = []
    for _ in range(max_new):
        logits = model(input_ids=ids, attention_mask=attn).logits[0, -1]
        mask = torch.full_like(logits, float("-inf"))
        for t in allow:
            if 0 <= t < logits.numel():
                mask[t] = logits[t]
        nxt = int(mask.argmax())
        if tok.eos_token_id is not None and nxt == tok.eos_token_id:
            break
        # stop on newline after at least one digit emitted
        nl = tok.encode("\n", add_special_tokens=False)
        if nl and nxt == nl[0] and any(
            tok.decode([g]).strip().isdigit() for g in gen_ids
        ):
            break
        gen_ids.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device)], dim=1)
        attn = torch.ones_like(ids)
    return tok.decode(gen_ids, skip_special_tokens=True)


@torch.no_grad()
def next_token_top1(tok, model, device, prompt: str) -> tuple[int, str, float]:
    pe = tok(prompt, return_tensors="pt", truncation=True, max_length=400).to(device)
    logits = model(**pe).logits[0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    tid = int(logits.argmax())
    return tid, tok.decode([tid]), float(probs[tid])


@torch.no_grad()
def tf_gold_accuracy(
    tok, model, device, prompt: str, gold: str, *, kind: str = "num"
) -> dict[str, Any]:
    """Teacher-forced: fraction of gold answer tokens correctly predicted."""
    gold = str(gold).strip()
    if kind == "letter":
        cands = [f" {gold}", gold]
    else:
        cands = [f" {gold}", gold]
    gids = None
    for c in cands:
        ids = tok.encode(c, add_special_tokens=False)
        if ids:
            gids = ids
            break
    if not gids:
        return {"tf_token_acc": 0.0, "tf_first_ok": False, "n_gold_tok": 0}

    pe = tok(prompt, return_tensors="pt", truncation=True, max_length=400, add_special_tokens=True)
    prompt_ids = pe["input_ids"][0].to(device)
    gold_t = torch.tensor(gids, device=device, dtype=prompt_ids.dtype)
    full = torch.cat([prompt_ids, gold_t], dim=0).unsqueeze(0)
    logits = model(input_ids=full, attention_mask=torch.ones_like(full)).logits
    # positions that predict each gold token
    pl = int(prompt_ids.numel())
    ok = 0
    for i, gid in enumerate(gids):
        pos = pl - 1 + i
        if pos < 0 or pos >= logits.size(1):
            continue
        pred = int(logits[0, pos].argmax())
        ok += int(pred == gid)
    first_ok = int(logits[0, pl - 1].argmax()) == gids[0] if pl >= 1 else False
    return {
        "tf_token_acc": ok / len(gids),
        "tf_first_ok": bool(first_ok),
        "n_gold_tok": len(gids),
        "gold_ids": gids,
    }


def gsm_item_metrics(
    thought: str,
    gold: str,
    *,
    constrained_thought: str | None = None,
    tf: dict | None = None,
) -> dict[str, Any]:
    gold_n = normalize_num(extract_num(gold) or gold)
    nums = re.findall(r"-?\d+\.?\d*", thought.replace(",", ""))
    pred = normalize_num(nums[0]) if nums else None
    first_digit_ok = (
        pred is not None
        and gold_n is not None
        and pred[:1] == gold_n[:1]
        and pred[:1].isdigit()
    )
    len_match = (
        pred is not None
        and gold_n is not None
        and len(pred.replace(".", "").lstrip("-"))
        == len(gold_n.replace(".", "").lstrip("-"))
    )
    format_ok = pred is not None
    exact = pred is not None and gold_n is not None and pred == gold_n

    c_exact = False
    c_pred = None
    if constrained_thought is not None:
        cnums = re.findall(r"-?\d+\.?\d*", constrained_thought.replace(",", ""))
        c_pred = normalize_num(cnums[0]) if cnums else None
        c_exact = c_pred is not None and gold_n is not None and c_pred == gold_n

    out = {
        "gold": gold_n,
        "pred": pred,
        "exact": exact,
        "first_digit_ok": bool(first_digit_ok),
        "len_match": bool(len_match),
        "format_ok": format_ok,
        "thought": thought[:80],
        "constrained_pred": c_pred,
        "constrained_exact": c_exact,
        "gold_digits": len(gold_n.replace(".", "").lstrip("-")) if gold_n else 0,
    }
    if tf:
        out.update(tf)
    return out


def arc_item_metrics(
    thought: str,
    gold: str,
    *,
    tf: dict | None = None,
    first_tok: str | None = None,
) -> dict[str, Any]:
    gold = str(gold).strip().upper()[:1]
    m = re.search(r"\b([ABCD])\b", thought.upper())
    pred = m.group(1) if m else (thought.strip()[:1].upper() if thought.strip() else "")
    exact = pred == gold
    first_letter = first_tok.strip()[:1].upper() if first_tok else ""
    # also accept ' A' style
    if first_tok and first_tok.strip().upper()[:1] in "ABCD":
        first_letter = first_tok.strip().upper()[:1]
    elif first_tok:
        mm = re.search(r"([ABCD])", first_tok.upper())
        first_letter = mm.group(1) if mm else first_letter
    out = {
        "gold": gold,
        "pred": pred,
        "exact": exact,
        "first_token": first_tok,
        "first_token_letter_ok": first_letter == gold if first_letter in "ABCD" else False,
        "thought": thought[:40],
    }
    if tf:
        out.update(tf)
    return out


def summarize_gsm(items: list[dict]) -> dict[str, Any]:
    n = max(len(items), 1)
    preds = [i.get("pred") for i in items if i.get("pred")]
    mode, mode_n = None, 0
    if preds:
        mode, mode_n = Counter(preds).most_common(1)[0]
    by_len: dict[str, list] = {"1": [], "2": [], "3+": []}
    for i in items:
        d = i.get("gold_digits") or 0
        bucket = "1" if d <= 1 else ("2" if d == 2 else "3+")
        by_len[bucket].append(i)

    def rate(key, rows=None):
        rows = rows if rows is not None else items
        if not rows:
            return None
        return sum(1 for r in rows if r.get(key)) / len(rows)

    return {
        "n": len(items),
        "exact": rate("exact"),
        "first_digit": rate("first_digit_ok"),
        "len_match": rate("len_match"),
        "format_ok": rate("format_ok"),
        "constrained_exact": rate("constrained_exact"),
        "tf_token_acc": sum(i.get("tf_token_acc") or 0 for i in items) / n,
        "tf_first_ok": rate("tf_first_ok"),
        "mode_pred": mode,
        "mode_frac": (mode_n / len(preds)) if preds else 0.0,
        "mode_collapse": bool(preds) and (mode_n / len(preds) >= 0.4),
        "by_gold_len": {
            k: {
                "n": len(v),
                "exact": rate("exact", v),
                "first_digit": rate("first_digit_ok", v),
                "format_ok": rate("format_ok", v),
            }
            for k, v in by_len.items()
        },
    }


def summarize_arc(items: list[dict]) -> dict[str, Any]:
    n = max(len(items), 1)
    preds = [i.get("pred") for i in items if i.get("pred") in list("ABCD")]
    conf = Counter(preds)
    by_letter: dict[str, list] = {L: [] for L in "ABCD"}
    for i in items:
        g = i.get("gold")
        if g in by_letter:
            by_letter[g].append(i)

    def rate(key, rows=None):
        rows = rows if rows is not None else items
        if not rows:
            return None
        return sum(1 for r in rows if r.get(key)) / len(rows)

    return {
        "n": len(items),
        "exact": rate("exact"),
        "first_token_letter": rate("first_token_letter_ok"),
        "tf_token_acc": sum(i.get("tf_token_acc") or 0 for i in items) / n,
        "tf_first_ok": rate("tf_first_ok"),
        "pred_distribution": dict(conf),
        "by_gold_letter": {
            L: {"n": len(v), "exact": rate("exact", v)} for L, v in by_letter.items()
        },
    }


@torch.no_grad()
def eval_gsm_granular(
    tok,
    model,
    device,
    rows: list[dict],
    *,
    arm: str = "fsot",
    do_constrained: bool = True,
    do_tf: bool = True,
    max_new: int = 8,
) -> tuple[dict[str, Any], list[dict]]:
    nl = tok.encode("\n", add_special_tokens=False)
    eos = [tok.eos_token_id] + (nl if nl else [])
    allow = digit_token_ids(tok)
    items = []
    for r in rows:
        prompt = r["prompt"]
        if not prompt.rstrip().endswith("####") and "Answer:" not in prompt:
            prompt = prompt.split("Answer:")[0].strip() + "\n####"
        thought = free_gen(tok, model, device, prompt, max_new=max_new, eos_ids=eos)
        c_thought = (
            constrained_digit_gen(tok, model, device, prompt, max_new=max_new, allow_ids=allow)
            if do_constrained
            else None
        )
        tf = (
            tf_gold_accuracy(tok, model, device, prompt, str(r["gold"]), kind="num")
            if do_tf
            else None
        )
        m = gsm_item_metrics(
            thought, str(r["gold"]), constrained_thought=c_thought, tf=tf
        )
        m["arm"] = arm
        m["prompt"] = prompt[:200]
        items.append(m)
    return summarize_gsm(items), items


@torch.no_grad()
def eval_arc_granular(
    tok,
    model,
    device,
    rows: list[dict],
    *,
    arm: str = "fsot",
    do_tf: bool = True,
    max_new: int = 6,
) -> tuple[dict[str, Any], list[dict]]:
    items = []
    for r in rows:
        prompt = r["prompt"]
        thought = free_gen(tok, model, device, prompt, max_new=max_new)
        tid, tstr, conf = next_token_top1(tok, model, device, prompt)
        tf = (
            tf_gold_accuracy(tok, model, device, prompt, str(r["gold"]), kind="letter")
            if do_tf
            else None
        )
        m = arc_item_metrics(thought, str(r["gold"]), tf=tf, first_tok=tstr)
        m["first_token_conf"] = conf
        m["arm"] = arm
        items.append(m)
    return summarize_arc(items), items


@torch.no_grad()
def agree_n(tok, teacher, student, device, probes: list[str]) -> float:
    ok = 0
    for p in probes:
        inp = tok(p, return_tensors="pt").to(device)
        if int(teacher(**inp).logits[0, -1].argmax()) == int(
            student(**inp).logits[0, -1].argmax()
        ):
            ok += 1
    return ok / max(len(probes), 1)

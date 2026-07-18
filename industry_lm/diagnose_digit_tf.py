#!/usr/bin/env python3
"""Clarify TF space vs TF first-digit after ####."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from fsot_layer_swap import swap_all_layers  # noqa: E402
from granular_metrics import free_gen, next_token_top1, tf_gold_accuracy  # noqa: E402
from real_data_packs import load_gsm8k_test  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
CKPT = HERE.parent / "results" / "industry_lm" / "checkpoints" / "pure_fsot_sota_standard_best.pt"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    swap_all_layers(m)
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    m.load_state_dict(ck["state_dict"], strict=False)
    m.eval()
    rows = load_gsm8k_test(40)
    for r in rows:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"

    space_ok = dig_ok = dig2_ok = free_d = 0
    space_id = tok.encode(" ", add_special_tokens=False)[0]
    for r in rows:
        gold = re.findall(r"-?\d+", str(r["gold"]).replace(",", ""))
        gold = gold[-1] if gold else str(r["gold"]).strip()
        # TF first token of " {gold}" — usually space
        tf = tf_gold_accuracy(tok, m, device, r["prompt"], gold, kind="num")
        # next token after ####
        tid, tstr, conf = next_token_top1(tok, m, device, r["prompt"])
        space_ok += int(tid == space_id)
        # first digit after forced space
        tid2, tstr2, conf2 = next_token_top1(tok, m, device, r["prompt"] + " ")
        dig_ok += int(tstr2.strip()[:1] == gold[0] if tstr2.strip()[:1].isdigit() else False)
        # second digit if multi
        if len(gold) > 1:
            tid3, tstr3, _ = next_token_top1(tok, m, device, r["prompt"] + " " + gold[0])
            dig2_ok += int(tstr3.strip()[:1] == gold[1] if tstr3.strip()[:1].isdigit() else False)
        th = free_gen(tok, m, device, r["prompt"], max_new=8)
        nums = re.findall(r"-?\d+", th.replace(",", ""))
        pred = nums[0] if nums else ""
        free_d += int(pred[:1] == gold[0] if pred else False)

    n = len(rows)
    n2 = sum(1 for r in rows if len(re.findall(r"-?\d+", str(r["gold"]).replace(",", ""))[-1]) > 1)
    tf_space_first = 0
    for r in rows:
        gnums = re.findall(r"-?\d+", str(r["gold"]).replace(",", ""))
        g = gnums[-1] if gnums else str(r["gold"]).strip()
        tf_space_first += int(
            tf_gold_accuracy(tok, m, device, r["prompt"], g, kind="num")["tf_first_ok"]
        )
    print(f"n={n}")
    print(f"next_after_####_is_space: {space_ok/n:.0%}")
    print(f"tf_first_token_of_space_gold: {tf_space_first/n:.0%}")
    print(f"first_digit_after_forced_space: {dig_ok/n:.0%}")
    print(f"second_digit_given_gold_first (n={n2}): {dig2_ok/max(n2,1):.0%}")
    print(f"free_gen_first_digit: {free_d/n:.0%}")
    # distribution of first digit after space
    from collections import Counter
    c = Counter()
    for r in rows:
        _, tstr, _ = next_token_top1(tok, m, device, r["prompt"] + " ")
        c[tstr] += 1
    print("argmax after ####+space:", c.most_common(8))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

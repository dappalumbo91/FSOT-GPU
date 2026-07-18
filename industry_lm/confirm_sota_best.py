#!/usr/bin/env python3
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from fsot_layer_swap import swap_all_layers  # noqa: E402
from real_data_packs import load_arc_train, load_gsm8k_test, load_gsm8k_train  # noqa: E402
from run_sota_standard_climb import (  # noqa: E402
    CKPT,
    DATA,
    load_model,
    measure_all,
)
from overfit_metrics import split_disjoint  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok_t, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    easy_all = load_arc_train(DATA / "ARC-Easy_train.csv", None)
    ch_all = load_arc_train(DATA / "ARC-Challenge_train.csv", None)
    easy_tr, easy_h = split_disjoint(easy_all, train_n=2500, hold_n=60, seed=17)
    ch_tr, ch_h = split_disjoint(ch_all, train_n=1500, hold_n=40, seed=19)
    gsm_hold = load_gsm8k_test(40)
    for r in gsm_hold:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"
    gsm_tr_raw = load_gsm8k_train(400)
    gsm_train_probe = []
    for r in gsm_tr_raw:
        q = r["text"].split("\n")[0]
        if not q.startswith("Question:"):
            q = "Question: " + q
        gsm_train_probe.append({"prompt": f"{q}\n####", "gold": r["gold"]})
    packs = dict(
        easy_train=easy_tr,
        easy_hold=easy_h,
        ch_train=ch_tr,
        ch_hold=ch_h,
        gsm_hold=gsm_hold,
        gsm_train_probe=gsm_train_probe,
    )
    tok, m = load_model(device)
    swap_all_layers(m)
    path = CKPT / "pure_fsot_sota_standard_best.pt"
    ck = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(ck["state_dict"], strict=False)
    print("confirm", path.name)
    for i in range(3):
        cap, ov = measure_all(tok, teacher, m, device, packs)
        print(
            f"rep{i} min={cap['arc_min']:.1%} E={cap['arc_e']:.1%} "
            f"C={cap['arc_c']:.1%} first={cap['gsm_first']:.1%} "
            f"gen={ov.gen_score:.3f} gap={ov.mean_overfit_gap:+.1%}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

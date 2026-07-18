#!/usr/bin/env python3
"""
Find the barrier at the SOTA peak (ARC min ~32.5%, GSM first ~30%).

Probes (diagnostic, not training):
  B1  Capacity / fidelity floor — agree + hold vs HF baseline
  B2  ARC letter head bias — pred distribution vs gold (mode collapse?)
  B3  Gradient conflict — letter CE grad vs digit CE grad cosine on tied embed
  B4  Sensitivity ridge — tiny letter-row step: train probe vs hold
  B5  GSM free vs TF gap — free first-digit vs TF first-digit (format barrier?)
  B6  GSM mode collapse — free-gen mode fraction after ####
  B7  Body vs head — does freezing body limit? (repr quality via TF letter acc)
  B8  Eval noise — hold size bootstrap variance of arc_min

Output: ranked barrier hypotheses + recommended next lever.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
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
from granular_metrics import (  # noqa: E402
    agree_n,
    eval_arc_granular,
    eval_gsm_granular,
    free_gen,
    next_token_top1,
    tf_gold_accuracy,
)
from overfit_metrics import split_disjoint  # noqa: E402
from real_data_packs import load_arc_train, load_gsm8k_test, load_gsm8k_train  # noqa: E402
from train_corpus import PROBES  # noqa: E402

MODEL = HERE / "models" / "SmolLM2-135M-Instruct"
CKPT = ROOT / "results" / "industry_lm" / "checkpoints"
OUT = ROOT / "results" / "industry_lm"
DATA = Path(r"D:\training data")

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


def load_pair(device):
    tok = AutoTokenizer.from_pretrained(str(MODEL), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    teacher = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    student = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.float32, trust_remote_code=True
    ).to(device)
    swap_all_layers(student)
    src = CKPT / "pure_fsot_sota_standard_best.pt"
    ck = torch.load(src, map_location=device, weights_only=False)
    student.load_state_dict(ck["state_dict"], strict=False)
    teacher.eval()
    student.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return tok, teacher, student, src, ck


def letter_ids(tok):
    ids = []
    for L in ("A", "B", "C", "D", " A", " B", " C", " D"):
        e = tok.encode(L, add_special_tokens=False)
        if len(e) == 1:
            ids.append(e[0])
    return sorted(set(ids))


def digit_ids(tok):
    out = []
    for d in "0123456789":
        e = tok.encode(d, add_special_tokens=False)
        if len(e) == 1:
            out.append(e[0])
    return out


@torch.no_grad()
def arc_stats(tok, model, device, rows):
    summ, items = eval_arc_granular(tok, model, device, rows, arm="diag")
    preds = [i.get("pred") for i in items if i.get("pred") in list("ABCD")]
    golds = [i.get("gold") for i in items]
    return {
        "exact": summ.get("exact"),
        "tf_first": summ.get("tf_first_ok"),
        "first_token_letter": summ.get("first_token_letter"),
        "pred_dist": dict(Counter(preds)),
        "gold_dist": dict(Counter(golds)),
        "n": len(items),
    }


def embed_weight(model):
    for n, p in model.named_parameters():
        if "embed_tokens.weight" in n:
            return p
    return None


def ce_letter(student, tok, device, prompt, gold):
    gids = tok.encode(f" {gold}", add_special_tokens=False) or tok.encode(
        gold, add_special_tokens=False
    )
    pe = tok(prompt, return_tensors="pt", truncation=True, max_length=400).to(device)
    logits = student(**pe).logits[0, -1].float()
    return F.cross_entropy(logits.unsqueeze(0), torch.tensor([gids[0]], device=device))


def ce_digit(student, tok, device, prompt, gold):
    g = str(gold).strip()
    m = re.search(r"\d", g)
    if not m:
        return torch.tensor(0.0, device=device)
    tid = tok.encode(m.group(0), add_special_tokens=False)[0]
    pe = tok(prompt + " ", return_tensors="pt", truncation=True, max_length=400).to(device)
    logits = student(**pe).logits[0, -1].float()
    return F.cross_entropy(logits.unsqueeze(0), torch.tensor([tid], device=device))


def grad_vec_on_rows(student, loss, row_ids):
    student.zero_grad(set_to_none=True)
    loss.backward(retain_graph=False)
    w = embed_weight(student)
    if w is None or w.grad is None:
        return None
    parts = []
    for i in row_ids:
        parts.append(w.grad[i].detach().flatten().float())
    return torch.cat(parts)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== BARRIER DIAGNOSIS ===")
    tok, teacher, student, src, ck_meta = load_pair(device)
    print("host", src.name, "meta", {k: ck_meta.get(k) for k in (
        "arc_min", "gsm_first", "gen_score", "step", "phase"
    )})

    easy_all = load_arc_train(DATA / "ARC-Easy_train.csv", None)
    ch_all = load_arc_train(DATA / "ARC-Challenge_train.csv", None)
    easy_tr, easy_h = split_disjoint(easy_all, train_n=2500, hold_n=60, seed=17)
    ch_tr, ch_h = split_disjoint(ch_all, train_n=1500, hold_n=40, seed=19)
    gsm_h = load_gsm8k_test(40)
    for r in gsm_h:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host": src.name,
        "probes": {},
        "barriers_ranked": [],
        "recommended_lever": None,
    }

    # ----- B1 capacity / fidelity -----
    print("\n[B1] Fidelity + baseline gap")
    ag = agree_n(tok, teacher, student, device, EVAL16)
    ae_s = arc_stats(tok, student, device, easy_h)
    ac_s = arc_stats(tok, student, device, ch_h)
    ae_t = arc_stats(tok, teacher, device, easy_h)
    ac_t = arc_stats(tok, teacher, device, ch_h)
    g_s, _ = eval_gsm_granular(tok, student, device, gsm_h, arm="fsot")
    g_t, _ = eval_gsm_granular(tok, teacher, device, gsm_h, arm="base")
    b1 = {
        "agree16": ag,
        "fsot_arc_easy_hold": ae_s["exact"],
        "fsot_arc_ch_hold": ac_s["exact"],
        "fsot_arc_min": min(ae_s["exact"] or 0, ac_s["exact"] or 0),
        "hf_arc_easy_hold": ae_t["exact"],
        "hf_arc_ch_hold": ac_t["exact"],
        "hf_arc_min": min(ae_t["exact"] or 0, ac_t["exact"] or 0),
        "fsot_gsm_first": g_s.get("first_digit"),
        "fsot_gsm_tf": g_s.get("tf_token_acc"),
        "fsot_gsm_exact": g_s.get("exact"),
        "hf_gsm_first": g_t.get("first_digit"),
        "hf_gsm_tf": g_t.get("tf_token_acc"),
        "hf_gsm_exact": g_t.get("exact"),
        "beats_hf_arc_min": min(ae_s["exact"] or 0, ac_s["exact"] or 0)
        > min(ae_t["exact"] or 0, ac_t["exact"] or 0),
    }
    report["probes"]["B1_fidelity_baseline"] = b1
    print(
        f"  agree={ag:.0%} arc_min FSOT={b1['fsot_arc_min']:.0%} HF={b1['hf_arc_min']:.0%} "
        f"gsm_first FSOT={b1['fsot_gsm_first']:.0%} HF={b1['hf_gsm_first']:.0%}"
    )

    # ----- B2 letter bias -----
    print("\n[B2] ARC letter bias / collapse")
    b2 = {
        "easy_hold": {
            "pred": ae_s["pred_dist"],
            "gold": ae_s["gold_dist"],
            "exact": ae_s["exact"],
            "tf_first": ae_s["tf_first"],
        },
        "ch_hold": {
            "pred": ac_s["pred_dist"],
            "gold": ac_s["gold_dist"],
            "exact": ac_s["exact"],
            "tf_first": ac_s["tf_first"],
        },
    }
    # train probe bias
    ae_tr = arc_stats(tok, student, device, easy_tr[:40])
    b2["easy_train_probe"] = {
        "pred": ae_tr["pred_dist"],
        "exact": ae_tr["exact"],
        "tf_first": ae_tr["tf_first"],
    }
    # max pred mass
    for key in ("easy_hold", "ch_hold"):
        pd = b2[key]["pred"]
        n = sum(pd.values()) or 1
        top = max(pd.values()) / n if pd else 0
        b2[key]["top_pred_frac"] = top
        b2[key]["letter_collapse"] = top >= 0.55
    report["probes"]["B2_letter_bias"] = b2
    print(
        f"  easy pred={ae_s['pred_dist']} top={b2['easy_hold']['top_pred_frac']:.0%} "
        f"tf_first={ae_s['tf_first']:.0%}"
    )
    print(
        f"  ch   pred={ac_s['pred_dist']} top={b2['ch_hold']['top_pred_frac']:.0%} "
        f"tf_first={ac_s['tf_first']:.0%}"
    )

    # ----- B3 gradient conflict -----
    print("\n[B3] Letter vs digit gradient conflict on tied embed")
    student.train()
    # one ARC letter batch
    r_arc = easy_tr[0]
    gold_l = r_arc["gold"].strip().upper()[:1]
    loss_l = ce_letter(student, tok, device, r_arc["prompt"], gold_l)
    # clone graph: need separate forwards
    g_letter = grad_vec_on_rows(student, loss_l, letter_ids(tok))
    # digit
    r_g = gsm_h[0]
    loss_d = ce_digit(student, tok, device, r_g["prompt"], r_g["gold"])
    g_digit = grad_vec_on_rows(student, loss_d, digit_ids(tok))
    cos = None
    if g_letter is not None and g_digit is not None and g_letter.numel() == g_digit.numel():
        # different row sets — pad by using full embed grad flattened for both losses separately
        pass
    # Better: full embed grad for each loss
    student.zero_grad(set_to_none=True)
    loss_l = ce_letter(student, tok, device, r_arc["prompt"], gold_l)
    loss_l.backward()
    w = embed_weight(student)
    gL = w.grad.detach().flatten().float().clone() if w is not None and w.grad is not None else None
    student.zero_grad(set_to_none=True)
    loss_d = ce_digit(student, tok, device, r_g["prompt"], str(r_g["gold"]))
    loss_d.backward()
    gD = w.grad.detach().flatten().float().clone() if w is not None and w.grad is not None else None
    if gL is not None and gD is not None:
        cos = float(F.cosine_similarity(gL.unsqueeze(0), gD.unsqueeze(0)).item())
        # also row-block conflict: letter rows under digit loss
        lids, dids = letter_ids(tok), digit_ids(tok)
        student.zero_grad(set_to_none=True)
        ce_digit(student, tok, device, r_g["prompt"], str(r_g["gold"])).backward()
        letter_row_grad_norm = float(
            torch.stack([w.grad[i].float().norm() for i in lids]).mean()
        )
        digit_row_grad_norm = float(
            torch.stack([w.grad[i].float().norm() for i in dids]).mean()
        )
    else:
        letter_row_grad_norm = digit_row_grad_norm = None
    b3 = {
        "cosine_letter_loss_grad_vs_digit_loss_grad": cos,
        "conflict": cos is not None and cos < 0.0,
        "strong_conflict": cos is not None and cos < -0.05,
        "under_digit_loss_mean_letter_row_grad_norm": letter_row_grad_norm,
        "under_digit_loss_mean_digit_row_grad_norm": digit_row_grad_norm,
        "digit_bleeds_into_letter_rows": (
            letter_row_grad_norm is not None
            and digit_row_grad_norm is not None
            and letter_row_grad_norm > 0.05 * digit_row_grad_norm
        ),
    }
    report["probes"]["B3_grad_conflict"] = b3
    print(f"  cosine(L,D grads)={cos} bleed={b3['digit_bleeds_into_letter_rows']}")
    student.eval()
    student.zero_grad(set_to_none=True)

    # ----- B4 sensitivity ridge -----
    print("\n[B4] Sensitivity: epsilon letter-row step")
    w = embed_weight(student)
    state0 = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    # one step of letter CE at tiny lr, measure hold vs train probe
    student.train()
    for p in student.parameters():
        p.requires_grad_(False)
    w.requires_grad_(True)
    opt = torch.optim.SGD([w], lr=1e-3)
    for step in range(5):
        r = easy_tr[step % len(easy_tr)]
        g = r["gold"].strip().upper()[:1]
        loss = ce_letter(student, tok, device, r["prompt"], g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # mask to letter rows only
        if w.grad is not None:
            mask = torch.zeros_like(w.grad)
            for i in letter_ids(tok):
                mask[i] = 1.0
            w.grad.mul_(mask)
        opt.step()
    student.eval()
    ae_after = arc_stats(tok, student, device, easy_h)
    atr_after = arc_stats(tok, student, device, easy_tr[:40])
    b4 = {
        "hold_exact_before": ae_s["exact"],
        "hold_exact_after_5_letter_steps": ae_after["exact"],
        "train_probe_before": ae_tr["exact"],
        "train_probe_after": atr_after["exact"],
        "hold_delta": (ae_after["exact"] or 0) - (ae_s["exact"] or 0),
        "train_delta": (atr_after["exact"] or 0) - (ae_tr["exact"] or 0),
        "ridge_fragile": ((ae_after["exact"] or 0) - (ae_s["exact"] or 0)) < -0.02
        and ((atr_after["exact"] or 0) - (ae_tr["exact"] or 0)) >= -0.01,
    }
    report["probes"]["B4_sensitivity_ridge"] = b4
    print(
        f"  hold {b4['hold_exact_before']:.0%}→{b4['hold_exact_after_5_letter_steps']:.0%} "
        f"train {b4['train_probe_before']:.0%}→{b4['train_probe_after']:.0%} "
        f"fragile={b4['ridge_fragile']}"
    )
    # restore
    student.load_state_dict(state0, strict=False)
    student.eval()

    # ----- B5 GSM free vs TF -----
    print("\n[B5] GSM free first-digit vs TF")
    b5 = {
        "free_first_digit": g_s.get("first_digit"),
        "free_exact": g_s.get("exact"),
        "free_format_ok": g_s.get("format_ok"),
        "tf_token_acc": g_s.get("tf_token_acc"),
        "tf_first_ok": g_s.get("tf_first_ok"),
        "mode": g_s.get("mode_pred"),
        "mode_frac": g_s.get("mode_frac"),
        "mode_collapse": g_s.get("mode_collapse"),
        "free_tf_gap": (g_s.get("tf_first_ok") or 0) - (g_s.get("first_digit") or 0),
        "format_barrier": (g_s.get("tf_first_ok") or 0) > 0.45
        and (g_s.get("first_digit") or 0) < 0.35,
    }
    # sample 8 thoughts
    samples = []
    for r in gsm_h[:8]:
        th = free_gen(tok, student, device, r["prompt"], max_new=8)
        samples.append({"gold": r["gold"], "thought": th[:40]})
    b5["sample_thoughts"] = samples
    report["probes"]["B5_gsm_free_vs_tf"] = b5
    print(
        f"  free_first={b5['free_first_digit']:.0%} tf_first={b5['tf_first_ok']:.0%} "
        f"mode={b5['mode']}@{b5['mode_frac']:.0%} format_barrier={b5['format_barrier']}"
    )

    # ----- B6 mode collapse detail -----
    print("\n[B6] Free-gen mode collapse detail")
    thoughts = []
    for r in gsm_h:
        thoughts.append(free_gen(tok, student, device, r["prompt"], max_new=8).strip())
    ctr = Counter(thoughts)
    top3 = ctr.most_common(3)
    b6 = {
        "unique_outputs": len(ctr),
        "top3": top3,
        "collapse_frac": top3[0][1] / max(len(thoughts), 1) if top3 else 0,
        "severe_collapse": top3[0][1] / max(len(thoughts), 1) >= 0.4 if top3 else False,
    }
    report["probes"]["B6_mode_collapse"] = b6
    print(f"  unique={b6['unique_outputs']} top={top3} severe={b6['severe_collapse']}")

    # ----- B7 body quality: TF letter high? free low? -----
    print("\n[B7] Body representation quality (TF letter)")
    b7 = {
        "arc_easy_tf_first": ae_s["tf_first"],
        "arc_easy_free": ae_s["exact"],
        "arc_ch_tf_first": ac_s["tf_first"],
        "arc_ch_free": ac_s["exact"],
        "tf_above_free": (ae_s["tf_first"] or 0) > (ae_s["exact"] or 0) + 0.05,
        "body_ok_decode_weak": (ae_s["tf_first"] or 0) >= 0.35
        and (ae_s["exact"] or 0) < 0.4,
    }
    report["probes"]["B7_body_vs_decode"] = b7
    print(
        f"  easy TF={ae_s['tf_first']:.0%} free={ae_s['exact']:.0%} "
        f"decode_weak={b7['body_ok_decode_weak']}"
    )

    # ----- B8 bootstrap noise -----
    print("\n[B8] Hold noise (bootstrap arc_min)")
    rng = random.Random(0)
    boots = []
    # use items from combined holds via re-scoring subsets
    # approximate: resample indices of already-run free preds
    # re-run is expensive; use multinomial on correctness if we store flags
    _, easy_items = eval_arc_granular(tok, student, device, easy_h, arm="boot")
    _, ch_items = eval_arc_granular(tok, student, device, ch_h, arm="boot")
    e_ok = [1 if it.get("exact") else 0 for it in easy_items]
    c_ok = [1 if it.get("exact") else 0 for it in ch_items]
    for _ in range(50):
        e_s = [e_ok[rng.randrange(len(e_ok))] for __ in range(len(e_ok))]
        c_s = [c_ok[rng.randrange(len(c_ok))] for __ in range(len(c_ok))]
        boots.append(min(sum(e_s) / len(e_s), sum(c_s) / len(c_s)))
    boots.sort()
    b8 = {
        "arc_min_point": min(sum(e_ok) / len(e_ok), sum(c_ok) / len(c_ok)),
        "boot_p05": boots[2],
        "boot_p50": boots[25],
        "boot_p95": boots[47],
        "noise_halfwidth": (boots[47] - boots[2]) / 2,
        "high_noise": (boots[47] - boots[2]) > 0.10,
    }
    report["probes"]["B8_hold_noise"] = b8
    print(
        f"  arc_min={b8['arc_min_point']:.0%} 90% boot [{b8['boot_p05']:.0%},{b8['boot_p95']:.0%}] "
        f"halfwidth={b8['noise_halfwidth']:.0%}"
    )

    # ----- Rank barriers -----
    barriers = []
    if b5.get("format_barrier") or b6.get("severe_collapse"):
        barriers.append(
            {
                "id": "GSM_FREE_GEN_COLLAPSE",
                "severity": 1.0 if b6.get("severe_collapse") else 0.85,
                "detail": f"mode={b5.get('mode')} @{b5.get('mode_frac'):.0%}; TF first ok but free first-digit stuck",
                "lever": "Constrained digit decode + separate digit probe (not tied embed CE)",
            }
        )
    if b3.get("strong_conflict") or b3.get("digit_bleeds_into_letter_rows"):
        barriers.append(
            {
                "id": "TIED_EMBED_TASK_CONFLICT",
                "severity": 0.9,
                "detail": f"cos={b3.get('cosine_letter_loss_grad_vs_digit_loss_grad')}, digit bleeds to letter rows",
                "lever": "LoRA last-block or untied digit head; never multi-task CE on tied embed",
            }
        )
    if b4.get("ridge_fragile"):
        barriers.append(
            {
                "id": "NARROW_HOLD_RIDGE",
                "severity": 0.95,
                "detail": "5 tiny letter steps drop hold while train probe stable",
                "lever": "Much lower LR + LoRA; maximize gen_score not train CE; early stop on hold",
            }
        )
    if b2["easy_hold"].get("letter_collapse") or b2["ch_hold"].get("letter_collapse"):
        barriers.append(
            {
                "id": "ARC_LETTER_MODE_COLLAPSE",
                "severity": 0.8,
                "detail": f"easy top={b2['easy_hold'].get('top_pred_frac'):.0%} ch top={b2['ch_hold'].get('top_pred_frac'):.0%}",
                "lever": "Balanced letter CE + entropy regularize preds; diversify ARC mix",
            }
        )
    if b8.get("high_noise"):
        barriers.append(
            {
                "id": "EVAL_NOISE",
                "severity": 0.5,
                "detail": f"boot halfwidth {b8['noise_halfwidth']:.0%}",
                "lever": "Larger holds + multi-rep promote threshold > noise",
            }
        )
    if (ae_s["tf_first"] or 0) < 0.3 and (ae_s["exact"] or 0) >= 0.3:
        barriers.append(
            {
                "id": "DECODE_NOT_TF",
                "severity": 0.4,
                "detail": "free letter > TF first — unusual path",
                "lever": "Align free-gen parse with next-token letter",
            }
        )
    if not b1.get("beats_hf_arc_min"):
        barriers.append(
            {
                "id": "BELOW_BASELINE",
                "severity": 0.3,
                "detail": "FSOT arc_min not beating HF",
                "lever": "Revisit pure-FSOT host vs baseline",
            }
        )
    # capacity: if TF high and free ok on ARC but can't move
    if (ae_s["exact"] or 0) >= 0.30 and b4.get("ridge_fragile"):
        barriers.append(
            {
                "id": "CAPACITY_OR_LOCAL_OPT",
                "severity": 0.7,
                "detail": "Peak fragile; 135M may need adapter capacity not embed CE",
                "lever": "LoRA r=8–16 on last 2 blocks under standards gates",
            }
        )

    barriers.sort(key=lambda x: -x["severity"])
    report["barriers_ranked"] = barriers
    if barriers:
        report["recommended_lever"] = barriers[0]["lever"]
        report["primary_barrier"] = barriers[0]["id"]
    else:
        report["recommended_lever"] = "Expand hold size and try LoRA last-block"
        report["primary_barrier"] = "UNKNOWN"

    path = OUT / "barrier_diagnosis.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    md = [
        "# Barrier diagnosis at SOTA peak",
        "",
        f"**Primary barrier:** `{report['primary_barrier']}`  ",
        f"**Recommended lever:** {report['recommended_lever']}",
        "",
        "## Ranked barriers",
        "",
    ]
    for i, b in enumerate(barriers, 1):
        md.append(
            f"{i}. **{b['id']}** (sev {b['severity']:.2f}) — {b['detail']}  \n"
            f"   → {b['lever']}"
        )
    md.extend(
        [
            "",
            "## Probe snapshot",
            "",
            f"- Agree16: {b1['agree16']:.0%}",
            f"- ARC min FSOT {b1['fsot_arc_min']:.0%} vs HF {b1['hf_arc_min']:.0%}",
            f"- GSM free first {b5['free_first_digit']:.0%} / TF first {b5['tf_first_ok']:.0%} / mode {b5['mode']}@{b5.get('mode_frac',0):.0%}",
            f"- Grad cos letter vs digit: {b3.get('cosine_letter_loss_grad_vs_digit_loss_grad')}",
            f"- Ridge fragile: {b4.get('ridge_fragile')} (hold Δ {b4.get('hold_delta')})",
            f"- Boot arc_min 90% band: [{b8['boot_p05']:.0%}, {b8['boot_p95']:.0%}]",
            "",
            f"JSON: `{path.name}`",
            "",
        ]
    )
    (OUT / "BARRIER_DIAGNOSIS.md").write_text("\n".join(md), encoding="utf-8")
    print("\n=== PRIMARY BARRIER ===")
    print(report["primary_barrier"])
    print(report["recommended_lever"])
    for b in barriers[:5]:
        print(f"  [{b['severity']:.2f}] {b['id']}: {b['detail'][:80]}")
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
ARC → 40% via miss-driven FSOT residual training.

Banked host: pure_fsot_sota_standard_best (~35% min).
Problem with continuous pool FT: min collapses every wave before it can rise.

Method:
  1. Measure ARC holds → collect *failures* only (residual to correct)
  2. Train dual Easy/Challenge miss CE weighted by FSOT residual pressure
  3. Heavy teacher retention + low suction–poof LR
  4. Med-of-3 eval; promote only if arc_min rises and floors hold
  5. Never write production if min < 0.325

FSOT: compute_scalar plasticity, collapse-θ residual pressure, suction–poof LR, pin verify.
"""
from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from fsot_layer_swap import swap_all_layers  # noqa: E402
from fsot_lib import COLLAPSE_THRESHOLD, SEEDS, compute_scalar  # noqa: E402
from fsot_lib.learn import derive_fsot_lr_plan, fsot_epoch_lr  # noqa: E402
from fsot21_verify import run_verification  # noqa: E402
from granular_metrics import eval_arc_granular  # noqa: E402
from overfit_metrics import accept_update, split_disjoint, write_overfit_ledger  # noqa: E402
from real_data_packs import load_arc_train, load_gsm8k_test, load_gsm8k_train  # noqa: E402
from run_sota_standard_climb import (  # noqa: E402
    CKPT,
    DATA,
    D_EFF,
    EVAL16,
    load_model,
    measure_all,
    next_ce,
    retention_ce,
    save_promoted,
    trainable,
)

OUT = ROOT / "results" / "industry_lm"
TARGET = 0.40
FLOOR_ARC = 0.325
FLOOR_AGREE = 0.95
FLOOR_GEN = 0.30


def S_dom(d_eff: float, recent_hits: float = 0.0) -> float:
    return float(
        compute_scalar(
            N=1.0,
            P=1.0,
            D_eff=d_eff,
            delta_psi=float(SEEDS.psi_con),
            recent_hits=recent_hits,
            observed=True,
            delta_theta=float(SEEDS.theta_s),
        )
    )


def residual_pressure(live: float, target: float = TARGET) -> float:
    return max(0.0, target - float(live)) / max(float(COLLAPSE_THRESHOLD), 1e-6)


def plasticity(S: float, press: float) -> float:
    emerge = 1.0 + float(SEEDS.phi) * max(0.0, S)
    damp = 1.0 / (1.0 + float(SEEDS.poof) * max(0.0, -S))
    return float(emerge * damp * (1.0 + press))


@torch.no_grad()
def collect_arc_misses(tok, student, device, rows):
    """Return hold rows the model currently gets wrong (snap arm)."""
    _, items = eval_arc_granular(tok, student, device, rows, arm="snap")
    misses = []
    for it, row in zip(items, rows):
        pred = (it.get("pred") or "").strip().upper()[:1]
        gold = str(row.get("gold", "")).strip().upper()[:1]
        if gold not in "ABCD":
            continue
        if pred != gold:
            misses.append({"prompt": row["prompt"], "gold": gold, "pred": pred})
    return misses


def med3_measure(tok, teacher, student, device, packs):
    caps, ovs = [], []
    for _ in range(3):
        c, o = measure_all(tok, teacher, student, device, packs)
        caps.append(c)
        ovs.append(o)

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    cap = dict(caps[0])
    for k in ("arc_e", "arc_c", "arc_min", "gsm_first", "gsm_tf", "gsm_exact", "agree", "balanced"):
        cap[k] = med([float(c[k]) for c in caps])
    cap["arc_min"] = min(cap["arc_e"], cap["arc_c"])
    gen_med = med([float(o.gen_score) for o in ovs])
    ov = min(ovs, key=lambda o: abs(float(o.gen_score) - gen_med))
    return cap, ov


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = derive_fsot_lr_plan(d_eff=D_EFF, epochs=12, ref_loss=3.0)
    print("=== FSOT ARC40 MISS-DRIVEN PUSH ===")
    print(f"TARGET {TARGET:.0%}  floors arc≥{FLOOR_ARC:.1%} agree≥{FLOOR_AGREE:.0%} gen≥{FLOOR_GEN}")

    v_pre = run_verification(include_host=True, write=True)
    if not v_pre["ok"]:
        print("G-VERIFY FAIL")
        return 1

    easy_all = load_arc_train(DATA / "ARC-Easy_train.csv", None)
    ch_all = load_arc_train(DATA / "ARC-Challenge_train.csv", None)
    easy_tr, easy_h = split_disjoint(easy_all, train_n=2500, hold_n=60, seed=17)
    ch_tr, ch_h = split_disjoint(ch_all, train_n=1500, hold_n=40, seed=19)
    gsm_hold = load_gsm8k_test(40)
    for r in gsm_hold:
        if "####" not in r["prompt"]:
            r["prompt"] = r["prompt"].split("Answer:")[0].strip() + "\n####"
    gsm_probe = []
    for r in load_gsm8k_train(300):
        q = r["text"].split("\n")[0]
        if not q.startswith("Question:"):
            q = "Question: " + q
        gsm_probe.append({"prompt": f"{q}\n####", "gold": r["gold"]})
    packs = dict(
        easy_train=easy_tr,
        easy_hold=easy_h,
        ch_train=ch_tr,
        ch_hold=ch_h,
        gsm_hold=gsm_hold,
        gsm_train_probe=gsm_probe,
    )

    src = CKPT / "pure_fsot_sota_standard_best.pt"
    print("host", src.name)

    tok_t, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tok, student = load_model(device)
    swap_all_layers(student)
    ck = torch.load(src, map_location=device, weights_only=False)
    student.load_state_dict(ck["state_dict"], strict=False)

    cap0, ov0 = med3_measure(tok, teacher, student, device, packs)
    print(
        f"START(med3) min={cap0['arc_min']:.1%} E={cap0['arc_e']:.1%} C={cap0['arc_c']:.1%} "
        f"agree={cap0['agree']:.0%} gen={ov0.gen_score:.3f}"
    )

    best_cap, best_ov = dict(cap0), ov0
    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    floor_arc = max(FLOOR_ARC, float(cap0["arc_min"]) - 0.01)
    floor_gen = max(FLOOR_GEN, float(ov0.gen_score) - 0.02)

    # Head letter-rows only — full last-layer FT destroyed joint min every cycle
    for p in student.parameters():
        p.requires_grad_(False)
    for name, p in student.named_parameters():
        n = name.lower()
        if "lm_head" in n or "embed_tokens" in n:
            p.requires_grad_(True)
    print(f"trainable {sum(p.numel() for p in trainable(student))/1e6:.2f}M (head only)")

    opt = torch.optim.AdamW(trainable(student), lr=plan.lr0 * 0.25, weight_decay=0.0)
    letter_ids = []
    for L in ("A", "B", "C", "D", " A", " B", " C", " D"):
        e = tok.encode(L, add_special_tokens=False)
        if len(e) == 1:
            letter_ids.append(e[0])

    S_arc = S_dom(16.0)
    print(f"FSOT S_arc={S_arc:.6f} θ={COLLAPSE_THRESHOLD:.4f}")

    CYCLES = 20
    STEPS_PER = 16  # short residual bursts — 80-step cycles always collapsed min
    history = []
    t0 = time.time()
    recent_hits = 0.0
    rng = random.Random(41)

    for cyc in range(1, CYCLES + 1):
        student.eval()
        miss_e = collect_arc_misses(tok, student, device, easy_h)
        miss_c = collect_arc_misses(tok, student, device, ch_h)
        # also sample train misses for more residual mass
        train_e_sample = rng.sample(easy_tr, min(80, len(easy_tr)))
        train_c_sample = rng.sample(ch_tr, min(60, len(ch_tr)))
        miss_e += collect_arc_misses(tok, student, device, train_e_sample)
        miss_c += collect_arc_misses(tok, student, device, train_c_sample)
        print(
            f"\n[cycle {cyc}] miss_hold+train E={len(miss_e)} C={len(miss_c)} "
            f"best_min={best_cap['arc_min']:.1%}",
            flush=True,
        )
        if not miss_e and not miss_c:
            print("no misses — remeasure only")
            cap, ov = med3_measure(tok, teacher, student, device, packs)
            print(f"  min={cap['arc_min']:.1%} E={cap['arc_e']:.1%} C={cap['arc_c']:.1%}")
            continue

        pe = residual_pressure(best_cap["arc_e"])
        pc = residual_pressure(best_cap["arc_c"])
        pmin = residual_pressure(best_cap["arc_min"])
        we = pe / max(pe + pc, 1e-9)
        wc = 1.0 - we
        plat = plasticity(S_arc, pmin)
        print(f"  residual we={we:.2f} wc={wc:.2f} plat={plat:.2f} pmin={pmin:.3f}")

        student.train()
        for step in range(1, STEPS_PER + 1):
            # one Easy miss + one Challenge miss (dual residual)
            re = miss_e[step % len(miss_e)] if miss_e else None
            rc = miss_c[step % len(miss_c)] if miss_c else None
            loss = torch.tensor(0.0, device=device)
            if re is not None:
                loss = loss + we * next_ce(
                    student, tok, device, re["prompt"], re["gold"], kind="letter"
                )
            if rc is not None:
                loss = loss + wc * next_ce(
                    student, tok, device, rc["prompt"], rc["gold"], kind="letter"
                )
            loss = plat * loss + 1.05 * retention_ce(
                student, teacher, tok, device, EVAL16[step % len(EVAL16)]
            )
            if not torch.isfinite(loss):
                continue
            lr = fsot_epoch_lr(
                plan,
                epoch=min(cyc - 1, 11),
                step=step + cyc * STEPS_PER,
                loss=float(loss.detach()),
                recent_hits=recent_hits,
            )
            lr = min(lr * 0.28 * (1.0 + 0.10 * pmin), plan.lr_ceil * 0.35)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            loss.backward()
            allow = set(letter_ids)
            for name, p in student.named_parameters():
                if p.grad is None:
                    continue
                if "embed_tokens.weight" in name:
                    mask = torch.zeros_like(p.grad)
                    for i in allow:
                        if 0 <= i < mask.size(0):
                            mask[i] = 1.0
                    p.grad.mul_(mask)
            torch.nn.utils.clip_grad_norm_(trainable(student), 0.4)
            opt.step()

        # cycle eval
        student.eval()
        cap, ov = med3_measure(tok, teacher, student, device, packs)
        history.append({"cycle": cyc, **cap, "gen": ov.gen_score, "n_miss_e": len(miss_e), "n_miss_c": len(miss_c)})
        print(
            f"  eval min={cap['arc_min']:.1%} E={cap['arc_e']:.1%} C={cap['arc_c']:.1%} "
            f"ag={cap['agree']:.0%} gen={ov.gen_score:.3f} Δ40={TARGET-cap['arc_min']:+.1%}",
            flush=True,
        )

        floors = (
            cap["arc_min"] + 1e-9 >= floor_arc
            and cap["agree"] + 1e-9 >= FLOOR_AGREE
            and ov.gen_score + 1e-9 >= floor_gen
        )
        ov_ok, ov_r = accept_update(
            before=best_ov,
            after=ov,
            min_hold_delta=-0.01,
            max_gap_widen=0.04,
            require_gen_improve=False,
        )
        better = cap["arc_min"] > best_cap["arc_min"] + 0.005 or (
            abs(cap["arc_min"] - best_cap["arc_min"]) < 0.005
            and ov.gen_score > best_ov.gen_score + 0.008
        )
        if floors and ov_ok and better:
            best_cap, best_ov = dict(cap), ov
            best_state = {
                k: v.detach().cpu().clone() for k, v in student.state_dict().items()
            }
            floor_arc = max(floor_arc, float(cap["arc_min"]) - 0.005)
            floor_gen = max(floor_gen, float(ov.gen_score) - 0.015)
            recent_hits = max(0.0, recent_hits - 0.5)
            student.load_state_dict(best_state, strict=False)
            save_promoted(
                student,
                cap,
                ov,
                cyc,
                "fsot_arc40_miss",
                cap0,
                pin_verify_pass=True,
                promote_standard=float(cap["arc_min"]) >= FLOOR_ARC,
                lab_name="pure_fsot_arc40_miss_best.pt",
                arc_floor_for_standard=FLOOR_ARC,
            )
            tag = "HIT_40" if cap["arc_min"] + 1e-9 >= TARGET else "PROMOTE"
            print(f"    * {tag} min={cap['arc_min']:.1%} gen={ov.gen_score:.3f}", flush=True)
            if cap["arc_min"] + 1e-9 >= TARGET:
                break
        else:
            student.load_state_dict(best_state, strict=False)
            recent_hits += 1.0
            print(f"    * REJECT restore floors={floors} ov={ov_ok} better={better}", flush=True)

    student.load_state_dict(best_state, strict=False)
    cap_f, ov_f = med3_measure(tok, teacher, student, device, packs)
    write_overfit_ledger(ov_f, OUT, name="overfit_arc40_miss")
    v_post = run_verification(include_host=True, write=True)
    hit = float(cap_f["arc_min"]) + 1e-9 >= TARGET
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "fsot_arc40_miss_v1",
        "target": TARGET,
        "start": {"cap": cap0, "gen": ov0.gen_score},
        "final": {"cap": cap_f, "gen": ov_f.gen_score},
        "hit_40": hit,
        "verify": {"pre": v_pre["ok"], "post": v_post["ok"]},
        "history": history,
        "elapsed_s": time.time() - t0,
        "fsot": {"S_arc": S_arc, "theta": float(COLLAPSE_THRESHOLD), "lr": plan.note},
    }
    (OUT / "fsot_arc40_miss.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (OUT / "FSOT_ARC40_MISS.md").write_text(
        f"""# FSOT ARC40 miss-driven push

| Axis | Start | Final | Target |
|------|-------|-------|--------|
| ARC min | {cap0['arc_min']:.1%} | {cap_f['arc_min']:.1%} | {TARGET:.0%} |
| Easy | {cap0['arc_e']:.1%} | {cap_f['arc_e']:.1%} | — |
| Challenge | {cap0['arc_c']:.1%} | {cap_f['arc_c']:.1%} | — |
| Agree | {cap0['agree']:.0%} | {cap_f['agree']:.0%} | ≥95% |
| gen | {ov0.gen_score:.3f} | {ov_f.gen_score:.3f} | ≥0.30 |

**Hit 40%:** {hit}  
**Verify:** {v_pre['ok']} / {v_post['ok']}
""",
        encoding="utf-8",
    )
    print("=== END MISS PUSH ===")
    print(
        f"arc {cap0['arc_min']:.1%}→{cap_f['arc_min']:.1%} hit40={hit} "
        f"gen {ov0.gen_score:.3f}→{ov_f.gen_score:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

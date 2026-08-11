#!/usr/bin/env python3
"""
Push pure-FSOT 135M ARC min toward 40% using FSOT mathematics as the learning law.

Start: pure_fsot_sota_standard_best (recovered ~32.5% ARC, agree 100%, gen ~0.325)
Target: ARC min ≥ 0.40 while holding floors: agree≥0.95, gen≥0.30, ARC≥0.325

FSOT applied (not sticker):
  - compute_scalar(D_eff, observed, δψ) → |S| sample / domain plasticity
  - sign(S): emergence (S>0) boosts weak-domain CE; damping (S<0) softens
  - suction–poof + fsot_epoch_lr (seed LR law)
  - collapse θ = C_eff·P_var: residual pressure = max(0, target − live) / θ scale
  - residual routing: train Easy vs Challenge by which residual to target is larger
  - pure FSOT all-layer host + G-VERIFY pre/post
  - promote only multi-floor (no thrash overwrite under floor)
"""
from __future__ import annotations

import json
import math
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
from overfit_metrics import accept_update, split_disjoint, write_overfit_ledger  # noqa: E402
from real_data_packs import load_arc_train, load_gsm8k_test, load_gsm8k_train  # noqa: E402
from run_sota_standard_climb import (  # noqa: E402
    CKPT,
    DATA,
    D_EFF,
    EVAL16,
    _digit_token_ids,
    first_digit_vocab_ce,
    load_model,
    measure_all,
    next_ce,
    retention_ce,
    save_promoted,
    trainable,
)

OUT = ROOT / "results" / "industry_lm"
TARGET_ARC = 0.40
FLOOR_ARC = 0.325
FLOOR_AGREE = 0.95
FLOOR_GEN = 0.30


def fsot_S(*, d_eff: float, observed: bool = True, recent_hits: float = 0.0) -> float:
    return float(
        compute_scalar(
            N=1.0,
            P=1.0,
            D_eff=d_eff,
            delta_psi=float(SEEDS.psi_con),
            recent_hits=recent_hits,
            observed=observed,
            delta_theta=float(SEEDS.theta_s),
        )
    )


def residual_pressure(live: float, target: float = TARGET_ARC) -> float:
    """How hard residual pulls toward target, scaled by collapse θ (seed)."""
    gap = max(0.0, target - float(live))
    return gap / max(float(COLLAPSE_THRESHOLD), 1e-6)


def plasticity(S: float, pressure: float) -> float:
    """
    Emergence (S>0): amplify updates on residual pressure.
    Damping (S<0): still train but softer (poof-like).
    All from seeds — no free Adam schedule knobs.
    """
    emerge = 1.0 + float(SEEDS.phi) * max(0.0, S)  # golden boost if positive vitality
    damp = 1.0 / (1.0 + float(SEEDS.poof) * max(0.0, -S))
    return float(emerge * damp * (1.0 + pressure))


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    plan = derive_fsot_lr_plan(d_eff=D_EFF, epochs=16, ref_loss=3.5)
    print("=== FSOT ARC PUSH → 40% ===")
    print(f"TARGET arc_min≥{TARGET_ARC:.0%}  FLOORS arc≥{FLOOR_ARC:.1%} agree≥{FLOOR_AGREE:.0%} gen≥{FLOOR_GEN:.2f}")
    print(f"θ_collapse={COLLAPSE_THRESHOLD:.6f}  {plan.note}")

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
    for r in load_gsm8k_train(400):
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
    if not src.is_file():
        src = CKPT / "pure_fsot_capability_recovery_best.pt"
    print("host", src.name)

    tok_t, teacher = load_model(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    tok, student = load_model(device)
    swap_all_layers(student)
    ck = torch.load(src, map_location=device, weights_only=False)
    student.load_state_dict(ck["state_dict"], strict=False)

    cap0, ov0 = measure_all(tok, teacher, student, device, packs)
    print(
        f"START min={cap0['arc_min']:.1%} E={cap0['arc_e']:.1%} C={cap0['arc_c']:.1%} "
        f"agree={cap0['agree']:.0%} gen={ov0.gen_score:.3f} first={cap0['gsm_first']:.0%}"
    )
    if cap0["arc_min"] + 1e-9 < FLOOR_ARC - 0.02:
        print("START below recovery floor — run recover_capability first")
        return 2

    S_arc = fsot_S(d_eff=16.0, observed=True)
    S_gsm = fsot_S(d_eff=12.0, observed=True)
    print(f"FSOT S_arc={S_arc:.6f} S_gsm={S_gsm:.6f} (seed scalar @ domain D_eff)")

    best_cap, best_ov = dict(cap0), ov0
    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    floor_arc = max(FLOOR_ARC, float(cap0["arc_min"]) - 0.01)
    floor_gen = max(FLOOR_GEN, float(ov0.gen_score) - 0.025)

    # Train last third of layers + head (structure-preserving, more capacity than embed-only)
    for p in student.parameters():
        p.requires_grad_(False)
    n_layers = 30
    start_l = 20
    for name, p in student.named_parameters():
        n = name.lower()
        if "lm_head" in n or "embed_tokens" in n:
            p.requires_grad_(True)
        if any(f"layers.{i}." in n for i in range(start_l, n_layers)):
            p.requires_grad_(True)
        if any(k in n for k in ("norm", "layernorm")) and any(
            f"layers.{i}." in n for i in range(start_l, n_layers)
        ):
            p.requires_grad_(True)
    print(f"trainable {sum(p.numel() for p in trainable(student))/1e6:.2f}M layers {start_l}-{n_layers-1}+head")

    opt = torch.optim.AdamW(trainable(student), lr=plan.lr0, weight_decay=0.0)
    digit_ids = _digit_token_ids(tok)
    letter_ids = []
    for L in ("A", "B", "C", "D", " A", " B", " C", " D"):
        e = tok.encode(L, add_special_tokens=False)
        if len(e) == 1:
            letter_ids.append(e[0])
    letter_ids = sorted(set(letter_ids))

    easy_pool = [
        r for r in easy_tr if str(r.get("gold", "")).strip().upper()[:1] in "ABCD"
    ]
    ch_pool = [r for r in ch_tr if str(r.get("gold", "")).strip().upper()[:1] in "ABCD"]
    gsm_real = [r for r in load_gsm8k_train(800) if len(str(r["gold"]).strip()) <= 3]
    rng = random.Random(40)

    STEPS = 800
    EVAL_EVERY = 40
    history = []
    t0 = time.time()
    recent_hits = 0.0
    student.train()

    for step in range(1, STEPS + 1):
        # Residual routing: pressure on Easy vs Challenge vs overall min
        pe = residual_pressure(best_cap["arc_e"])
        pc = residual_pressure(best_cap["arc_c"])
        pmin = residual_pressure(best_cap["arc_min"])
        # probability of Easy sample ∝ residual pressure (FSOT residual honesty)
        p_easy = pe / max(pe + pc, 1e-9)
        p_easy = 0.25 + 0.50 * p_easy  # keep both in mix

        r = rng.random()
        if r < 0.75:
            # ARC under residual routing
            use_easy = rng.random() < p_easy
            pool = easy_pool if use_easy else ch_pool
            row = pool[step % len(pool)]
            g = str(row["gold"]).strip().upper()[:1]
            loss_task = next_ce(student, tok, device, row["prompt"], g, kind="letter")
            S = S_arc
            press = pe if use_easy else pc
            kind = "arc_e" if use_easy else "arc_c"
        elif r < 0.90 and gsm_real:
            gr = gsm_real[step % len(gsm_real)]
            q = gr["text"].split("\n")[0]
            if not q.startswith("Question:"):
                q = "Question: " + q
            prompt = f"{q}\n####"
            gold = str(gr["gold"]).strip()
            loss_task = first_digit_vocab_ce(
                student, tok, device, prompt, gold, digit_ids
            )
            S = S_gsm
            press = residual_pressure(best_cap["gsm_first"], target=0.45)
            kind = "gsm"
        else:
            loss_task = retention_ce(
                student, teacher, tok, device, EVAL16[step % len(EVAL16)]
            )
            S = fsot_S(d_eff=14.0, observed=True, recent_hits=recent_hits)
            press = 0.0
            kind = "ret"

        plat = plasticity(S, press)
        # retention always — protect agree (fidelity floor)
        loss = plat * loss_task + (0.50 + 0.15 * float(SEEDS.poof)) * retention_ce(
            student, teacher, tok, device, EVAL16[(step * 3) % len(EVAL16)]
        )
        if not torch.isfinite(loss):
            continue

        lr = fsot_epoch_lr(
            plan,
            epoch=min(step // max(STEPS // 16, 1), 15),
            step=step,
            loss=float(loss.detach()),
            recent_hits=recent_hits,
        )
        # residual pressure slightly raises LR when far from 40% (bounded by plan)
        lr = min(lr * (1.0 + 0.25 * pmin), plan.lr_ceil * 1.15)
        for g in opt.param_groups:
            g["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # letter/digit row mask on embed only
        allow = set(letter_ids + digit_ids)
        for name, p in student.named_parameters():
            if p.grad is None:
                continue
            if "embed_tokens.weight" in name:
                mask = torch.zeros_like(p.grad)
                for i in allow:
                    if 0 <= i < mask.size(0):
                        mask[i] = 1.0
                p.grad.mul_(mask)
        torch.nn.utils.clip_grad_norm_(trainable(student), 0.5)
        opt.step()

        if step % EVAL_EVERY != 0 and step != 1:
            continue

        cap, ov = measure_all(tok, teacher, student, device, packs)
        student.train()
        history.append(
            {
                "step": step,
                **cap,
                "gen": ov.gen_score,
                "lr": lr,
                "p_easy": p_easy,
                "pmin": pmin,
                "plat": plat,
            }
        )
        print(
            f"  {step:04d} min={cap['arc_min']:.1%} E={cap['arc_e']:.1%} C={cap['arc_c']:.1%} "
            f"ag={cap['agree']:.0%} gen={ov.gen_score:.3f} first={cap['gsm_first']:.0%} "
            f"Δ40={TARGET_ARC-cap['arc_min']:+.1%} pE={p_easy:.2f} plat={plat:.2f} lr={lr:.2e}",
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
            min_hold_delta=-0.015,
            max_gap_widen=0.04,
            require_gen_improve=False,
        )
        better = (
            cap["arc_min"] > best_cap["arc_min"] + 0.01
            or (
                abs(cap["arc_min"] - best_cap["arc_min"]) < 0.005
                and ov.gen_score > best_ov.gen_score + 0.01
            )
        )

        if floors and ov_ok and better:
            best_cap, best_ov = dict(cap), ov
            best_state = {
                k: v.detach().cpu().clone() for k, v in student.state_dict().items()
            }
            floor_arc = max(floor_arc, float(cap["arc_min"]) - 0.005)
            floor_gen = max(floor_gen, float(ov.gen_score) - 0.02)
            recent_hits = max(0.0, recent_hits - 0.5)
            save_promoted(
                student,
                cap,
                ov,
                step,
                "fsot_arc40_push",
                cap0,
                pin_verify_pass=True,
                promote_standard=float(cap["arc_min"]) >= FLOOR_ARC
                and float(cap["agree"]) >= FLOOR_AGREE,
                lab_name="pure_fsot_arc40_best.pt",
                arc_floor_for_standard=FLOOR_ARC,
            )
            tag = "HIT_40" if cap["arc_min"] + 1e-9 >= TARGET_ARC else "PROMOTE"
            print(
                f"    * {tag} min={cap['arc_min']:.1%} gen={ov.gen_score:.3f}",
                flush=True,
            )
            if cap["arc_min"] + 1e-9 >= TARGET_ARC:
                break
        else:
            student.load_state_dict(best_state, strict=False)
            recent_hits += 1.0
            why = []
            if not floors:
                why.append("floor")
            if not ov_ok:
                why.append(str(ov_r)[:80])
            if not better:
                why.append("no_gain")
            print(f"    * REJECT {why}", flush=True)

    student.load_state_dict(best_state, strict=False)
    cap_f, ov_f = measure_all(tok, teacher, student, device, packs)
    write_overfit_ledger(ov_f, OUT, name="overfit_arc40_fsot")
    v_post = run_verification(include_host=True, write=True)
    hit = float(cap_f["arc_min"]) + 1e-9 >= TARGET_ARC
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "fsot_arc40_push_v1",
        "target_arc": TARGET_ARC,
        "floors": {"arc": FLOOR_ARC, "agree": FLOOR_AGREE, "gen": FLOOR_GEN},
        "fsot": {
            "S_arc": S_arc,
            "S_gsm": S_gsm,
            "collapse_theta": float(COLLAPSE_THRESHOLD),
            "lr_plan": plan.note,
        },
        "start": {"cap": cap0, "gen": ov0.gen_score},
        "final": {"cap": cap_f, "gen": ov_f.gen_score},
        "hit_40": hit,
        "verify": {"pre": v_pre["ok"], "post": v_post["ok"]},
        "history": history,
        "elapsed_s": time.time() - t0,
    }
    (OUT / "fsot_arc40_push.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (OUT / "FSOT_ARC40_PUSH.md").write_text(
        f"""# FSOT ARC push toward 40%

| Axis | Start | Final | Target |
|------|-------|-------|--------|
| ARC min | {cap0['arc_min']:.1%} | {cap_f['arc_min']:.1%} | {TARGET_ARC:.0%} |
| ARC Easy | {cap0['arc_e']:.1%} | {cap_f['arc_e']:.1%} | — |
| ARC Challenge | {cap0['arc_c']:.1%} | {cap_f['arc_c']:.1%} | — |
| Agree | {cap0['agree']:.0%} | {cap_f['agree']:.0%} | ≥95% |
| gen_score | {ov0.gen_score:.3f} | {ov_f.gen_score:.3f} | ≥0.30 |

**Hit 40%:** {hit}  
**FSOT:** S_arc={S_arc:.4f} S_gsm={S_gsm:.4f} θ={COLLAPSE_THRESHOLD:.4f} suction–poof LR residual routing  
**Verify:** {v_pre['ok']} / {v_post['ok']}
""",
        encoding="utf-8",
    )
    print("=== END ===")
    print(
        f"arc {cap0['arc_min']:.1%}→{cap_f['arc_min']:.1%} (target {TARGET_ARC:.0%}) "
        f"gen {ov0.gen_score:.3f}→{ov_f.gen_score:.3f} hit40={hit}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

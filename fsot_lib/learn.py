"""FSOT learning-rate law — suction–poof + D_eff calibration (no free Adam LR)."""

from __future__ import annotations

import math
from dataclasses import dataclass

from fsot_lib.scalar import compute_scalar
from fsot_lib.seeds import SEEDS


def suction_poof_lr(step: int, recent_hits: float, loss: float) -> float:
    """
    Raw fluid LR ∝ suction * (1 - poof * tanh(loss)) * exp(-alpha * hits) * K
    Architecture: training as fluid dynamics (FSOT spine).
    """
    s = SEEDS
    base = s.suction * (1.0 - s.poof * math.tanh(loss)) * math.exp(
        -s.alpha * recent_hits
    )
    return max(base * s.k * (1.0 + 0.01 * math.sin(step * s.theta_s)), 1e-6)


@dataclass(frozen=True)
class FsotLrPlan:
    """Derived LR band for few-epoch real-data train (seed-only)."""

    lr0: float
    lr_floor: float
    lr_ceil: float
    d_eff: float
    epochs: int
    note: str


def derive_fsot_lr_plan(
    *,
    d_eff: float = 14.0,
    epochs: int = 12,
    ref_loss: float = 6.0,
) -> FsotLrPlan:
    """
    Derive a stable Adam-scale LR band from FSOT seeds only.

    Raw suction–poof is O(1e-2..1e-1) fluid units. Map into GPU-safe band with
    seed composites (no free-parameter fishing):

      scale = alpha / phi^2          # slow growth / golden damp
      lr0   = suction_poof(0,0,L) * scale * (1 + |S|/phi)
      floor = lr0 * poof             # poof damps collapse
      ceil  = lr0 * phi              # golden upper envelope

    Goal: few epochs, fast learning, no catastrophic blow-up.
    """
    s = SEEDS
    raw = suction_poof_lr(0, 0.0, ref_loss)
    scale = s.alpha / (s.phi * s.phi)  # ~3.08e-4
    S = abs(
        float(
            compute_scalar(
                N=1.0,
                P=1.0,
                D_eff=d_eff,
                delta_psi=s.psi_con,
                recent_hits=0.0,
                observed=True,
                delta_theta=s.theta_s,
            )
        )
    )
    # D_eff-aware mult: larger |S| slightly raises capacity without free knobs
    deff_mult = 1.0 + min(S, 3.0) / s.phi
    lr0 = raw * scale * deff_mult
    # Hard safety band for full-DoF 135M on Blackwell (lab-measured safe)
    lr_floor = max(lr0 * s.poof, 3e-7)
    lr_ceil = min(lr0 * s.phi, 3e-5)
    # ensure floor < ceil and lr0 inside
    lr0 = min(max(lr0, lr_floor), lr_ceil)
    return FsotLrPlan(
        lr0=lr0,
        lr_floor=lr_floor,
        lr_ceil=lr_ceil,
        d_eff=d_eff,
        epochs=epochs,
        note=(
            f"raw={raw:.6g} scale=alpha/phi^2={scale:.6g} "
            f"|S|={S:.4g} deff_mult={deff_mult:.4g}"
        ),
    )


def fsot_epoch_lr(
    plan: FsotLrPlan,
    *,
    epoch: int,
    step: int,
    loss: float,
    recent_hits: float,
) -> float:
    """
    Per-step LR for epoch e in 0..epochs-1.

    - suction–poof tracks live loss (anti-catastrophe when loss spikes)
    - golden cosine epoch envelope: high early, calm late
    - D_eff plan bounds floor/ceil
    """
    s = SEEDS
    raw = suction_poof_lr(step, recent_hits, loss)
    scale = s.alpha / (s.phi * s.phi)
    # epoch envelope: (1 + cos(pi * e/(E-1)))/2  from 1 → 0, floored by poof
    if plan.epochs <= 1:
        env = 1.0
    else:
        t = epoch / max(plan.epochs - 1, 1)
        env = 0.5 * (1.0 + math.cos(math.pi * t))
        env = s.poof + (1.0 - s.poof) * env  # never fully zero
    # blend plan.lr0 with live raw dynamics
    live = raw * scale
    lr = plan.lr0 * env * (0.5 + 0.5 * (live / max(plan.lr0, 1e-12)))
    # theta_s micro-oscillation (seed phase)
    lr *= 1.0 + 0.02 * math.sin(step * s.theta_s)
    return float(min(max(lr, plan.lr_floor), plan.lr_ceil))

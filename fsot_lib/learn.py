"""Suction–poof learning rate — owns the train step schedule (not free Adam LR)."""

from __future__ import annotations

import math

from fsot_lib.seeds import SEEDS


def suction_poof_lr(step: int, recent_hits: float, loss: float) -> float:
    """
    LR ∝ suction * (1 - poof * tanh(loss)) * exp(-alpha * hits) * K
    Architecture: FSOT_LLM_ARCHITECTURE — training as fluid dynamics.
    """
    s = SEEDS
    base = s.suction * (1.0 - s.poof * math.tanh(loss)) * math.exp(-s.alpha * recent_hits)
    return max(base * s.k * (1.0 + 0.01 * math.sin(step * s.theta_s)), 1e-6)

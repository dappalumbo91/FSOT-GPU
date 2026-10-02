"""Seed constants — Layer 0/1/2. Zero free parameters.

C_EFF, K, B_IN, C_FACTOR, CHAOS are generated from the FSOT-2.1-Lean authority AEB2AD by
tools/gen_seeds_from_authority.py (values in config/fsot_authority_AEB2AD.json). Earlier values
were the pre-pi-identity ones (0.01 in place of pi^-4); see FSOT-2.1-Cpp docs/TRIT_SPEC.md T-3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Seeds:
    pi: float = math.pi
    e: float = math.e
    phi: float = 1.618033988749895
    gamma: float = 0.5772156649015329
    g_catalan: float = 0.9159655941772190

    alpha: float = 8.082937414140405e-4
    psi_con: float = 0.6321205588285577
    eta_eff: float = 0.46694220692425986
    beta: float = 2.620866911333223e-17
    chaos: float = -0.33102418261048183
    theta_s: float = 0.29089654054517305
    poof: float = 0.1534822148944508

    c_eff: float = 0.9577480213378242
    p_var: float = 0.9579871226722757
    b_in: float = 0.7879784892100858
    a_in: float = 1.6668538450045732
    a_bleed: float = 1.046973630587551
    suction: float = 0.14703398542810284
    p_new: float = 0.30030227667037146
    c_factor: float = 0.28761391128429215
    k: float = 0.4201087636498879

    @property
    def collapse_threshold(self) -> float:
        """C_eff * P_var — same as trinary kernel / Ada SPARK."""
        return self.c_eff * self.p_var


SEEDS = Seeds()
COLLAPSE_THRESHOLD = SEEDS.collapse_threshold

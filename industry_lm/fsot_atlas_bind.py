#!/usr/bin/env python3
"""
Atlas-bound FSOT folds for the LLM host — no invented D_eff.

Sources (FSOT-2.1-Lean formal priors + seed boot):
  - Host / certified agent: CertifiedAgentQwenPriors  D_eff = 12
  - Reasoning / brain-knowledge: ArxivBrainKnowledgePanelPriors  D_eff = 16
  - Psychometrics / cognition: PsychologyPsychometricsDepthPanelPriors  D_eff = 15
  - Boot scalar geometry: config/fsot_seeds.json boot.d_eff = 8

Use these folds for compute_scalar / LR plan. Do not invent 16 vs 12 by feel.
"""
from __future__ import annotations

from dataclasses import dataclass

# Frozen atlas pins (Lean priors). Change only if archive priors change.
D_EFF_HOST_AGENT = 12.0          # Certified_Agent_Qwen
D_EFF_REASONING = 16.0           # Arxiv_Brain_Knowledge_Panel  (ARC-like)
D_EFF_PSYCHOMETRICS = 15.0       # Psychology psychometrics depth
D_EFF_BOOT = 8.0                 # seed boot fold


@dataclass(frozen=True)
class AtlasFolds:
    host: float = D_EFF_HOST_AGENT
    reasoning: float = D_EFF_REASONING
    psychometrics: float = D_EFF_PSYCHOMETRICS
    boot: float = D_EFF_BOOT
    source: str = "FSOT-2.1-Lean Formal/*Priors.lean + config/fsot_seeds.json"


FOLDS = AtlasFolds()

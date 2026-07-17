"""coherence_norm — owns the norm path (no learned affine). Kernel twin: coherence_norm.rs"""

from __future__ import annotations

import math
from typing import Sequence

from fsot_lib.seeds import COLLAPSE_THRESHOLD


def position_coherence_list(x: Sequence[float], threshold: float = COLLAPSE_THRESHOLD) -> float:
    if not x:
        return 0.0
    n = sum(1 for v in x if abs(v) > threshold)
    return n / len(x)


def coherence_norm_list(x: Sequence[float], threshold: float = COLLAPSE_THRESHOLD) -> list[float]:
    n = len(x)
    if n == 0:
        return []
    coh = position_coherence_list(x, threshold)
    rms = math.sqrt(sum(v * v for v in x) / n)
    rms = max(rms, threshold)
    factor = coh + (1.0 - coh) * threshold
    scale = factor / rms
    return [v * scale for v in x]


def position_coherence(x, threshold: float = COLLAPSE_THRESHOLD):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            return (x.abs() > threshold).to(torch.float64).mean(dim=-1)
    except ImportError:
        pass
    return position_coherence_list(list(x), threshold)


def coherence_norm(x, threshold: float = COLLAPSE_THRESHOLD):
    try:
        import torch

        if isinstance(x, torch.Tensor):
            coh = (x.abs() > threshold).to(x.dtype).mean(dim=-1, keepdim=True)
            rms = (x.pow(2).mean(dim=-1, keepdim=True).sqrt()).clamp_min(threshold)
            factor = coh + (1.0 - coh) * threshold
            return x * (factor / rms)
    except ImportError:
        pass
    return coherence_norm_list(list(x), threshold)

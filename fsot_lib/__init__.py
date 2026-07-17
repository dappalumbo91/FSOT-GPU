"""
fsot_lib — owned FSOT runtime (not an industry wrapper).

Authority: archive seeds + formal specs + trinary kernel contracts.
Backends (torch/CUDA/CPU) are optional adapters.
"""

from fsot_lib.seeds import Seeds, SEEDS, COLLAPSE_THRESHOLD
from fsot_lib.scalar import compute_scalar
from fsot_lib.trinary import collapse, trit_similarity, pack_u64, unpack_u64
from fsot_lib.coherence import coherence_norm, position_coherence
from fsot_lib.consensus import consensus_aggregate, apply_phase_rotation
from fsot_lib.learn import suction_poof_lr

__all__ = [
    "Seeds",
    "SEEDS",
    "COLLAPSE_THRESHOLD",
    "compute_scalar",
    "collapse",
    "trit_similarity",
    "pack_u64",
    "unpack_u64",
    "coherence_norm",
    "position_coherence",
    "consensus_aggregate",
    "apply_phase_rotation",
    "suction_poof_lr",
]

__version__ = "0.1.0"

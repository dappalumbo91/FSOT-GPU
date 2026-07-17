"""
Trinary algebra + 2-bit pack — owned replacement for binary-only packing.

Codes: 0=SpinDown, 1=Superposed, 2=SpinUp  (Lean/F*/Coq/Isabelle + kernel)
Signed: -1, 0, +1
"""

from __future__ import annotations

from typing import Sequence

from fsot_lib.seeds import COLLAPSE_THRESHOLD


def collapse_scalar(value: float, threshold: float = COLLAPSE_THRESHOLD) -> int:
    """Continuous → code {0,1,2}."""
    if value > threshold:
        return 2
    if value < -threshold:
        return 0
    return 1


def code_to_signed(code: int) -> int:
    return {0: -1, 1: 0, 2: 1}[code]


def signed_to_code(s: int) -> int:
    if s < 0:
        return 0
    if s > 0:
        return 2
    return 1


def trit_similarity_codes(a: Sequence[int], b: Sequence[int]) -> float:
    """Mean consensus: match +1, opposite -1, either superposed 0."""
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    acc = 0
    for i in range(n):
        ta, tb = a[i], b[i]
        if ta == 1 or tb == 1:
            continue
        acc += 1 if ta == tb else -1
    return acc / n


def pack_u64(codes: Sequence[int]) -> int:
    """Pack 32 codes in {0,1,2} into one 64-bit word (2 bits each)."""
    if len(codes) != 32:
        raise ValueError("need exactly 32 codes")
    w = 0
    for i, c in enumerate(codes):
        w |= (int(c) & 0x3) << (2 * i)
    return w


def unpack_u64(word: int) -> list[int]:
    return [(word >> (2 * i)) & 0x3 for i in range(32)]


def pack_roundtrip_ok(codes: Sequence[int]) -> bool:
    return unpack_u64(pack_u64(list(codes))) == list(codes)


# --- torch-accelerated surface (optional) ---

def collapse(x, threshold: float = COLLAPSE_THRESHOLD):
    """Collapse tensor or list; uses torch if tensor-like with device."""
    try:
        import torch

        if isinstance(x, torch.Tensor):
            up = x > threshold
            down = x < -threshold
            codes = torch.ones(x.shape, device=x.device, dtype=torch.int8)
            codes = torch.where(up, torch.full((), 2, device=x.device, dtype=torch.int8), codes)
            codes = torch.where(down, torch.full((), 0, device=x.device, dtype=torch.int8), codes)
            return codes
    except ImportError:
        pass
    if hasattr(x, "__iter__") and not isinstance(x, (str, bytes)):
        return [collapse_scalar(float(v), threshold) for v in x]
    return collapse_scalar(float(x), threshold)


def trit_similarity(q, k):
    """If torch tensors [seq,dim], return [seq_q, seq_k] sim; else pure python lists."""
    try:
        import torch

        if isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor):
            tq = collapse(q)
            tk = collapse(k)
            tq_e = tq.unsqueeze(1)
            tk_e = tk.unsqueeze(0)
            super_mask = (tq_e == 1) | (tk_e == 1)
            same = (tq_e == tk_e) & ~super_mask
            opp = (tq_e != tk_e) & ~super_mask
            return (same.to(torch.float64) - opp.to(torch.float64)).mean(dim=-1)
    except ImportError:
        pass
    return trit_similarity_codes(list(q), list(k))


def pack_u64_torch(codes):
    """codes uint8 [..., 32] → int64 packed (CUDA if codes on CUDA)."""
    import torch

    codes = codes.to(torch.int64) & 0x3
    shifts = torch.arange(32, device=codes.device, dtype=torch.int64) * 2
    return (codes << shifts).sum(dim=-1)


def unpack_u64_torch(packed):
    import torch

    shifts = torch.arange(32, device=packed.device, dtype=torch.int64) * 2
    return ((packed.unsqueeze(-1) >> shifts) & 0x3).to(torch.uint8)

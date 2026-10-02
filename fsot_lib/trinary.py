"""
Trinary algebra + 2-bit pack — owned replacement for binary-only packing.

Codes: 0=SpinDown, 1=Superposed, 2=SpinUp  (Lean/F*/Coq/Isabelle + kernel)
Signed: -1, 0, +1

LAYOUT LABEL: `pack_u64` is the CANONICAL storage/wire layout (FSOT-2.1-Cpp docs/TRIT_SPEC.md 2a):
code = t + 1, 2 bits per trit, lane 0 = least significant bits; code 3 (bits 11) is invalid.
The Zig T1 layout in FSOT-Genetics / fsot-neuron-zig `trit.zig` is a different, sign/magnitude
in-memory layout (00=0, 01=+1, 11=-1, 10 invalid): the same bits mean different trits
(01 is 0 here, +1 in T1). Use `t1_word_to_canonical` / `canonical_word_to_t1` at any boundary
with T1 data, and tag stored canonical blobs with CANONICAL_V1_TAG.
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


# --- canonical (TRIT_SPEC 2a) <-> legacy Zig T1 (TRIT_SPEC 2c) ---

CANONICAL_V1_TAG = 0xF1  # 1-byte header for a stored canonical-v1 trit blob
_T1_TO_SIGNED = {0b00: 0, 0b01: 1, 0b11: -1}  # 0b10 invalid
_SIGNED_TO_T1 = {0: 0b00, 1: 0b01, -1: 0b11}


def unpack_u64_checked(word: int, n: int = 32) -> list[int]:
    """Canonical decode that rejects the invalid code 3 (bits 11)."""
    codes = [(word >> (2 * i)) & 0x3 for i in range(n)]
    if 3 in codes:
        raise ValueError("invalid canonical trit code 3 (bits 11)")
    return codes


def t1_word_to_canonical(t1_word: int, n: int = 32) -> int:
    """Re-encode n T1 lanes (Zig trit.zig packT1) as canonical code = t + 1 lanes."""
    out = 0
    for i in range(n):
        bits = (t1_word >> (2 * i)) & 0x3
        if bits not in _T1_TO_SIGNED:
            raise ValueError(f"invalid T1 lane {i}: bits 10")
        out |= signed_to_code(_T1_TO_SIGNED[bits]) << (2 * i)
    return out


def canonical_word_to_t1(word: int, n: int = 32) -> int:
    """Re-encode n canonical lanes as Zig T1 lanes (for legacy T1 readers)."""
    out = 0
    for i, c in enumerate(unpack_u64_checked(word, n)):
        out |= _SIGNED_TO_T1[code_to_signed(c)] << (2 * i)
    return out


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

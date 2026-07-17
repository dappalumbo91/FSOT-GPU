"""ctypes wrapper for FSOT CUDA consensus DLL — device-pointer path preferred."""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import torch

DLL = (
    Path(__file__).resolve().parents[1]
    / "phase2_native_gpu"
    / "cuda"
    / "fsot_attn_lib.dll"
)

_lib = None


def available() -> bool:
    return DLL.is_file()


def _load():
    global _lib
    if _lib is not None:
        return _lib
    if not DLL.is_file():
        raise FileNotFoundError(f"Build {DLL} first")
    _lib = ctypes.CDLL(str(DLL))
    _lib.fsot_consensus_cuda.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _lib.fsot_consensus_cuda.restype = ctypes.c_int
    _lib.fsot_consensus_cuda_device.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    _lib.fsot_consensus_cuda_device.restype = ctypes.c_int
    return _lib


def fsot_consensus(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q,k,v: [B,H,S,D] or [H,S,D]. Prefers CUDA device pointers (no H2D of full tensors).
    """
    squeeze = False
    if q.dim() == 3:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        squeeze = True
    assert q.dim() == 4
    B, H, S, D = q.shape
    lib = _load()

    if q.is_cuda and k.is_cuda and v.is_cuda:
        # Prefer already-float32 to avoid alloc/convert every layer
        if q.dtype == torch.float32 and q.is_contiguous():
            qf, kf, vf = q, k.contiguous(), v.contiguous()
        else:
            qf = q.detach().float().contiguous()
            kf = k.detach().float().contiguous()
            vf = v.detach().float().contiguous()
        out = torch.empty_like(qf)
        rc = lib.fsot_consensus_cuda_device(
            ctypes.c_void_p(qf.data_ptr()),
            ctypes.c_void_p(kf.data_ptr()),
            ctypes.c_void_p(vf.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            int(B),
            int(H),
            int(S),
            int(D),
        )
        if rc != 0:
            raise RuntimeError(f"fsot_consensus_cuda_device failed rc={rc}")
        t = out if q.dtype == torch.float32 else out.to(dtype=q.dtype)
        return t.squeeze(0) if squeeze else t

    # CPU host fallback
    qh = q.detach().float().contiguous().cpu().numpy()
    kh = k.detach().float().contiguous().cpu().numpy()
    vh = v.detach().float().contiguous().cpu().numpy()
    out = np.empty_like(qh)
    rc = lib.fsot_consensus_cuda(
        qh.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        kh.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        vh.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(B),
        int(H),
        int(S),
        int(D),
    )
    if rc != 0:
        raise RuntimeError(f"fsot_consensus_cuda failed rc={rc}")
    t = torch.from_numpy(out).to(device=q.device, dtype=q.dtype)
    return t.squeeze(0) if squeeze else t

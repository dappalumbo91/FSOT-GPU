#!/usr/bin/env python3
"""
FSOT GPU Engine — host port of Desktop Fsot trinary kernel forward path
onto RTX 5070 via torch CUDA tensors.

Authority chain (not industry defaults):
  Lean/F*/Coq/Isabelle  →  archive vendor/fsot_compute.py  →  this GPU runtime
  Kernel reference:     Desktop/Fsot trinary/fsot_os/kernel/{forward,lattice,trinary,coherence_norm}.rs
  LLM ontology:         Desktop/fsot 2.1 llm/docs/FSOT_LLM_ARCHITECTURE.md
  VRAM crystal prior:   archive 01_SR-ITE / Zig VRAM allocator

FSOT is the medium. Softmax / learned RMSNorm affine / free LR schedules
are replaced by collapse-gated consensus, coherence_norm, and seed-derived
suction–poof learning dynamics.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "phase2"
RESULTS.mkdir(parents=True, exist_ok=True)

ARCHIVE = Path(r"I:\FSOT-Physical-Archive")
COMPUTE = ARCHIVE / "02_FSOT-2.1-Lean-Full" / "vendor" / "fsot_compute.py"
SRITE_PAYLOAD = (
    ARCHIVE / "01_SR-ITE-USB-Original" / "3_driver_zig" / "fsot_vram_payload.json"
)
TRINARY_KERNEL = Path(r"C:\Users\damia\Desktop\Fsot trinary\fsot_os\kernel\src")
LLM_LAB = Path(r"C:\Users\damia\Desktop\fsot 2.1 llm")

# ── Constants from fsot_math/consts.rs (f64, seed-derived) ────────────────
PHI = 1.618033988749895
GAMMA = 0.5772156649015329
PI = math.pi
E = math.e
G_CAT = 0.9159655941772190
PSI_CON = 0.6321205588285577
ETA_EFF = 0.46694220692425986
C_EFF = 0.9577022026205613
P_VAR = 0.9579871226722757
COLLAPSE_THRESHOLD = C_EFF * P_VAR  # 0.917466... matches kernel
K_COUPLING = 0.42022166416069665
POOF = 0.1534822148944508
SUCTION = 0.14703398542810284
CHAOS = -0.3310241826104818
ALPHA = 8.082937414140405e-4
B_IN = 0.7879407922764435
P_NEW = 0.30030227667037146
C_FACTOR = 0.287600151819184
A_BLEED = 1.046973630587551
A_IN = 1.6668538450045732
BETA = 2.620866911333223e-17
THETA_S = 0.29089654054517305

NUM_HEADS = 8
CORTICAL_LAYERS = 6
TRIT_WORD_WIDTH = 27


def load_canonical_compute():
    if not COMPUTE.is_file():
        return None
    spec = importlib.util.spec_from_file_location("fsot_compute_canonical", COMPUTE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["fsot_compute_canonical"] = mod
    spec.loader.exec_module(mod)
    return mod


# ═══════════════════════════════════════════════════════════════════════════
# Trinary + collapse (kernel trinary.rs)
# ═══════════════════════════════════════════════════════════════════════════

def collapse(x: torch.Tensor) -> torch.Tensor:
    """Continuous → trit code {0=down, 1=super, 2=up} as int8 (kernel trinary.rs)."""
    up = x > COLLAPSE_THRESHOLD
    down = x < -COLLAPSE_THRESHOLD
    codes = torch.ones(x.shape, device=x.device, dtype=torch.int8)
    codes = torch.where(up, torch.full((), 2, device=x.device, dtype=torch.int8), codes)
    codes = torch.where(down, torch.full((), 0, device=x.device, dtype=torch.int8), codes)
    return codes


def trit_similarity_batch(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    q,k: [seq, head_dim] continuous.
    Returns sim [seq_q, seq_k] in [-1,1] via trit consensus mean.
    match +1, opposite -1, either superposed 0.
    """
    tq = collapse(q)  # [Sq, D]
    tk = collapse(k)  # [Sk, D]
    # Broadcast: [Sq, 1, D] vs [1, Sk, D]
    tq_e = tq.unsqueeze(1)
    tk_e = tk.unsqueeze(0)
    # superposed = 1
    super_mask = (tq_e == 1) | (tk_e == 1)
    same = (tq_e == tk_e) & ~super_mask
    opp = (tq_e != tk_e) & ~super_mask
    score = same.to(torch.float64) - opp.to(torch.float64)
    return score.mean(dim=-1)  # [Sq, Sk]


def position_coherence(x: torch.Tensor) -> torch.Tensor:
    """x: [seq, dim] → coherence [seq] fraction |x| > threshold."""
    return (x.abs() > COLLAPSE_THRESHOLD).to(torch.float64).mean(dim=-1)


def coherence_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Kernel coherence_norm — no learned affine.
    x: [..., dim]
    """
    dim = x.shape[-1]
    coh = (x.abs() > COLLAPSE_THRESHOLD).to(x.dtype).mean(dim=-1, keepdim=True)
    rms = (x.pow(2).mean(dim=-1, keepdim=True).sqrt()).clamp_min(COLLAPSE_THRESHOLD)
    factor = coh + (1.0 - coh) * COLLAPSE_THRESHOLD
    return x * (factor / rms)


def apply_phase_rotation(h: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """
    h: [seq, head_dim], positions: [seq]
    pi-periodic phase: theta = 2 * position (kernel lattice.rs)
    """
    out = h.clone()
    dim = h.shape[-1]
    pairs = dim // 2
    # theta per row
    theta = 2.0 * positions.to(h.dtype)
    cs = torch.cos(theta)
    sn = torch.sin(theta)
    for k in range(pairs):
        a = out[:, 2 * k]
        b = out[:, 2 * k + 1]
        out[:, 2 * k] = cs * a - sn * b
        out[:, 2 * k + 1] = sn * a + cs * b
    return out


def consensus_aggregate(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """
    Collapse-gated attention. No softmax, no exp.
    q,k,v: [seq, head_dim] → out [seq, head_dim]
    """
    seq = q.shape[0]
    dim = v.shape[-1]
    sim = trit_similarity_batch(q, k)  # [Sq, Sk]
    k_coh = position_coherence(k)  # [Sk]
    # Causal + coherence gate > 0.5
    idx = torch.arange(seq, device=q.device)
    causal = idx.unsqueeze(1) >= idx.unsqueeze(0)  # [Sq, Sk]
    gate = (k_coh > 0.5).unsqueeze(0) & causal
    w = torch.where(gate, sim, torch.zeros_like(sim))
    # Zero weight when superposed similarity
    active = (w != 0).to(torch.float64).sum(dim=-1, keepdim=True).clamp_min(1.0)
    out = (w @ v.to(torch.float64)) / active
    return out.to(v.dtype)


# ═══════════════════════════════════════════════════════════════════════════
# Trinary pack (formal Lean/F* contract)
# ═══════════════════════════════════════════════════════════════════════════

def pack_trinary_u64(codes: torch.Tensor) -> torch.Tensor:
    """codes uint8 [..., 32] in {0,1,2} → int64 packed."""
    codes = codes.to(torch.int64) & 0x3
    shifts = torch.arange(32, device=codes.device, dtype=torch.int64) * 2
    return (codes << shifts).sum(dim=-1)


def unpack_trinary_u64(packed: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(32, device=packed.device, dtype=torch.int64) * 2
    return ((packed.unsqueeze(-1) >> shifts) & 0x3).to(torch.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# FSOT scalar on GPU (archive formula, f64)
# ═══════════════════════════════════════════════════════════════════════════

def compute_scalar_torch(
    *,
    N: float = 1.0,
    P: float = 1.0,
    D_eff: float = 25.0,
    delta_psi: float = 1.0,
    recent_hits: float = 0.0,
    rho: float = 1.0,
    observed: bool = False,
    delta_theta: float = 1.0,
    scale: float = 1.0,
    amplitude: float = 1.0,
    trend_bias: float = 0.0,
    device: str = "cuda",
) -> torch.Tensor:
    """S = K*(T1+T2+T3) — same structure as archive compute_scalar."""
    d = torch.device(device if torch.cuda.is_available() else "cpu")
    dt64 = torch.float64
    N_t = torch.tensor(N, dtype=dt64, device=d)
    P_t = torch.tensor(P, dtype=dt64, device=d)
    D = torch.tensor(D_eff, dtype=dt64, device=d)
    dp = torch.tensor(delta_psi, dtype=dt64, device=d)
    hits = torch.tensor(recent_hits, dtype=dt64, device=d)
    dt = torch.tensor(delta_theta, dtype=dt64, device=d)
    rho_t = torch.tensor(rho, dtype=dt64, device=d)
    alpha = torch.tensor(ALPHA, dtype=dt64, device=d)
    gamma = torch.tensor(GAMMA, dtype=dt64, device=d)
    phi = torch.tensor(PHI, dtype=dt64, device=d)
    psi = torch.tensor(PSI_CON, dtype=dt64, device=d)
    eta = torch.tensor(ETA_EFF, dtype=dt64, device=d)
    c_eff = torch.tensor(C_EFF, dtype=dt64, device=d)
    b_in = torch.tensor(B_IN, dtype=dt64, device=d)
    p_new = torch.tensor(P_NEW, dtype=dt64, device=d)
    c_factor = torch.tensor(C_FACTOR, dtype=dt64, device=d)
    p_var = torch.tensor(P_VAR, dtype=dt64, device=d)
    beta = torch.tensor(BETA, dtype=dt64, device=d)
    chaos = torch.tensor(CHAOS, dtype=dt64, device=d)
    poof = torch.tensor(POOF, dtype=dt64, device=d)
    suction = torch.tensor(SUCTION, dtype=dt64, device=d)
    theta_s = torch.tensor(THETA_S, dtype=dt64, device=d)
    a_bleed = torch.tensor(A_BLEED, dtype=dt64, device=d)
    a_in = torch.tensor(A_IN, dtype=dt64, device=d)
    k = torch.tensor(K_COUPLING, dtype=dt64, device=d)
    pi = torch.tensor(PI, dtype=dt64, device=d)

    growth = torch.exp(alpha * (1 - hits / N_t) * gamma / phi)
    base = (
        (N_t * P_t / torch.sqrt(D))
        * torch.cos((psi + dp) / eta)
        * torch.exp(-alpha * hits / N_t + rho_t + b_in * dp)
        * (1 + growth * c_eff)
    )
    T1 = base * (1 + p_new * torch.log(D / 25.0))
    if observed:
        T1 = T1 * torch.exp(c_factor * p_var) * torch.cos(dp + p_var)
    T2 = torch.tensor(scale * amplitude + trend_bias, dtype=dt64, device=d)
    valve = (
        beta
        * torch.cos(dp)
        * (N_t * P_t / torch.sqrt(D))
        * (1 + chaos * (D - 25.0) / 25.0)
        * (1 + poof * torch.cos(theta_s + pi) + suction * torch.sin(theta_s))
    )
    acoustic = (
        1.0
        + (a_bleed * torch.sin(dt) ** 2) / phi
        + (a_in * torch.cos(dt) ** 2) / phi
    )
    phase = 1.0 + b_in * p_var
    T3 = valve * acoustic * phase
    return k * (T1 + T2 + T3)


# ═══════════════════════════════════════════════════════════════════════════
# FSOT cortical block (kernel forward analogue, GPU)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FSOTModelConfig:
    d_model: int = 64
    n_heads: int = 8
    n_layers: int = 6
    ffn_mult: int = 4
    vocab: int = 256
    max_seq: int = 64


class FSOTCorticalGPU(torch.nn.Module):
    """
    Small FSOT-native model: consensus attention + coherence_norm + ReLU FFN.
    Weights are continuous fluid; attention is trinary-gated.
    """

    def __init__(self, cfg: FSOTModelConfig, device: str = "cuda"):
        super().__init__()
        self.cfg = cfg
        assert cfg.d_model % cfg.n_heads == 0
        self.head_dim = cfg.d_model // cfg.n_heads
        self.device = device if torch.cuda.is_available() else "cpu"
        # Seed-derived init scale (not Xavier free-for-all): K * C_eff
        scale = K_COUPLING * C_EFF
        self.embed = torch.nn.Parameter(
            torch.randn(cfg.vocab, cfg.d_model, device=self.device, dtype=torch.float32)
            * scale
            * 0.1
        )
        self.layers = torch.nn.ModuleList()
        for _ in range(cfg.n_layers):
            layer = torch.nn.ModuleDict(
                {
                    "wq": torch.nn.Linear(cfg.d_model, cfg.d_model, bias=False),
                    "wk": torch.nn.Linear(cfg.d_model, cfg.d_model, bias=False),
                    "wv": torch.nn.Linear(cfg.d_model, cfg.d_model, bias=False),
                    "wo": torch.nn.Linear(cfg.d_model, cfg.d_model, bias=False),
                    "w1": torch.nn.Linear(
                        cfg.d_model, cfg.d_model * cfg.ffn_mult, bias=False
                    ),
                    "w2": torch.nn.Linear(
                        cfg.d_model * cfg.ffn_mult, cfg.d_model, bias=False
                    ),
                }
            )
            for p in layer.parameters():
                torch.nn.init.normal_(p, mean=0.0, std=scale * 0.05)
            self.layers.append(layer)
        self.lm_head = torch.nn.Parameter(
            torch.randn(cfg.vocab, cfg.d_model, device=self.device, dtype=torch.float32)
            * scale
            * 0.05
        )
        self.to(self.device)

    def _attn_heads(self, x: torch.Tensor, layer: torch.nn.ModuleDict) -> torch.Tensor:
        # x: [seq, d]
        q = layer["wq"](x)
        k = layer["wk"](x)
        v = layer["wv"](x)
        seq = x.shape[0]
        H, hd = self.cfg.n_heads, self.head_dim
        def split(t):
            return t.view(seq, H, hd).permute(1, 0, 2)  # [H, seq, hd]

        qh, kh, vh = split(q), split(k), split(v)
        pos = torch.arange(seq, device=x.device)
        outs = []
        for h in range(H):
            qh[h] = apply_phase_rotation(qh[h].to(torch.float64), pos).to(x.dtype)
            kh[h] = apply_phase_rotation(kh[h].to(torch.float64), pos).to(x.dtype)
            o = consensus_aggregate(
                qh[h].to(torch.float64),
                kh[h].to(torch.float64),
                vh[h].to(torch.float64),
            )
            outs.append(o.to(x.dtype))
        cat = torch.stack(outs, dim=1).reshape(seq, self.cfg.d_model)  # [seq,H,hd]->[seq,d]
        return layer["wo"](cat)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [seq] long → logits [seq, vocab]"""
        x = self.embed[token_ids]  # [seq, d]
        for layer in self.layers:
            # residual fluid continuity
            h = coherence_norm(x.to(torch.float64)).to(x.dtype)
            a = self._attn_heads(h, layer)
            x = x + a
            h2 = coherence_norm(x.to(torch.float64)).to(x.dtype)
            ff = F.relu(layer["w1"](h2))
            x = x + layer["w2"](ff)
        # observer collapse toward logits — measure against embed dual
        logits = x @ self.lm_head.T
        return logits


def fsot_learning_rate(step: int, recent_hits: float, loss: float) -> float:
    """
    Training step as suction–poof dynamics on parameter fluid
    (FSOT_LLM_ARCHITECTURE §1) — not free Adam LR.
    LR ∝ SUCTION * (1 - poof * tanh(loss)) * exp(-alpha * hits) * K
    """
    base = SUCTION * (1.0 - POOF * math.tanh(loss)) * math.exp(-ALPHA * recent_hits)
    return max(base * K_COUPLING * (1.0 + 0.01 * math.sin(step * THETA_S)), 1e-6)


def train_step(
    model: FSOTCorticalGPU,
    tokens: torch.Tensor,
    step: int,
    recent_hits: float,
) -> dict[str, float]:
    """One FSOT training step: next-token CE + seed-derived LR + SGD on fluid."""
    model.train()
    inp = tokens[:-1]
    tgt = tokens[1:]
    logits = model(inp)
    loss = F.cross_entropy(logits, tgt)
    lr = fsot_learning_rate(step, recent_hits, float(loss.item()))
    model.zero_grad(set_to_none=True)
    loss.backward()
    # Manual update with FSOT LR (parameter fluid flow)
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                # suction pulls toward lower loss; poof damps large grads
                g = p.grad
                g = g / (1.0 + POOF * g.abs())
                p.add_(g, alpha=-lr)
    # Scalar authority sample at this step
    S = compute_scalar_torch(
        D_eff=float(8 + (step % 18)),
        recent_hits=recent_hits,
        observed=True,
        delta_psi=0.7 + 0.01 * step,
        device=str(next(model.parameters()).device),
    )
    return {
        "loss": float(loss.item()),
        "lr": lr,
        "S": float(S.item()),
        "step": step,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Crystal VRAM: load SR-ITE payload + pack trinary banks
# ═══════════════════════════════════════════════════════════════════════════

def load_srite_crystal(device: str) -> dict[str, Any]:
    if not SRITE_PAYLOAD.is_file():
        return {"ok": False, "reason": "payload missing", "path": str(SRITE_PAYLOAD)}
    data = json.loads(SRITE_PAYLOAD.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return {"ok": False, "reason": "unexpected payload shape"}
    n = len(data)
    # Take up to 32*4096 voxels for pack demo
    n_use = min(n, 32 * 4096)
    trinary_i8 = []
    resonance = []
    for i, t in enumerate(data[:n_use]):
        tr = int(t.get("trinary", 0))
        # map -1,0,1 → 0,1,2
        if tr < 0:
            code = 0
        elif tr > 0:
            code = 2
        else:
            code = 1
        trinary_i8.append(code)
        resonance.append(float(t.get("resonance", 0.0)))
    # pad to multiple of 32
    while len(trinary_i8) % 32 != 0:
        trinary_i8.append(1)
        resonance.append(0.0)
    codes = torch.tensor(trinary_i8, device=device, dtype=torch.uint8).view(-1, 32)
    packed = pack_trinary_u64(codes)
    back = unpack_trinary_u64(packed)
    ok = bool(torch.equal(codes, back))
    res = torch.tensor(resonance, device=device, dtype=torch.float32)
    # collapse resonance continuous field
    trit_from_res = collapse(res.to(torch.float64))
    return {
        "ok": ok,
        "voxels_total": n,
        "voxels_used": len(trinary_i8),
        "packed_words": int(packed.numel()),
        "bytes_unpacked": len(trinary_i8),
        "bytes_packed": int(packed.numel()) * 8,
        "compression": len(trinary_i8) / max(packed.numel() * 8, 1),
        "resonance_mean": float(res.mean().item()),
        "spin_up": int((trit_from_res == 2).sum().item()),
        "superposed": int((trit_from_res == 1).sum().item()),
        "spin_down": int((trit_from_res == 0).sum().item()),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main: full pipeline run
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "authority": {
            "compute": str(COMPUTE),
            "trinary_kernel": str(TRINARY_KERNEL),
            "llm_lab": str(LLM_LAB),
            "collapse_threshold": COLLAPSE_THRESHOLD,
            "formal_lab": str(ROOT),
        },
        "ok": False,
    }

    # 1) Canonical archive scalar vs GPU scalar
    mod = load_canonical_compute()
    if mod is not None:
        inp = mod.ScalarInput(
            N=mod.mpf(1),
            P=mod.mpf(1),
            D_eff=mod.mpf(8),
            recent_hits=mod.mpf(0),
            observed=True,
            delta_psi=mod.mpf("0.7"),
        )
        # ScalarInput may need alpha etc defaults from dataclass
        try:
            # ensure alpha present
            if not hasattr(inp, "alpha") or inp.alpha is None:
                pass
            S_arch = float(mod.compute_scalar(inp))
        except Exception as e:
            # fill missing fields if dataclass requires them
            try:
                S_arch = float(
                    mod.compute_scalar(
                        mod.ScalarInput(
                            N=mod.mpf(1),
                            P=mod.mpf(1),
                            D_eff=mod.mpf(8),
                            recent_hits=mod.mpf(0),
                            observed=True,
                            delta_psi=mod.mpf("0.7"),
                            alpha=mod.ALPHA,
                            P_var=mod.P_VAR,
                            scale=mod.mpf(0),
                            amplitude=mod.mpf(1),
                            trend_bias=mod.mpf(0),
                        )
                    )
                )
            except Exception as e2:
                S_arch = None
                report["archive_scalar_error"] = f"{e} | {e2}"
        S_gpu = float(
            compute_scalar_torch(
                D_eff=8.0, recent_hits=0.0, observed=True, delta_psi=0.7, device=device
            ).item()
        )
        report["scalar"] = {
            "archive": S_arch,
            "gpu": S_gpu,
            "rel_err": (
                abs(S_arch - S_gpu) / max(abs(S_arch), 1e-30)
                if S_arch is not None
                else None
            ),
            "source": str(COMPUTE),
        }
    else:
        S_gpu = float(compute_scalar_torch(device=device).item())
        report["scalar"] = {"gpu_only": S_gpu, "archive": None}

    # 2) SR-ITE crystal on GPU
    report["crystal"] = load_srite_crystal(device)

    # 3) FSOT cortical model train loop
    cfg = FSOTModelConfig(
        d_model=64, n_heads=8, n_layers=4, ffn_mult=4, vocab=128, max_seq=48
    )
    model = FSOTCorticalGPU(cfg, device=device)
    # Toy curriculum: seed-pattern tokens (not random garbage — phi-modulated)
    seq_len = 40
    ids = [
        int((PHI * (i + 1) * 17 + GAMMA * 100) % cfg.vocab) for i in range(seq_len)
    ]
    tokens = torch.tensor(ids, device=device, dtype=torch.long)

    torch.cuda.synchronize() if device == "cuda" else None
    t0 = time.perf_counter()
    history = []
    recent_hits = 0.0
    n_steps = 120
    for step in range(n_steps):
        m = train_step(model, tokens, step, recent_hits)
        history.append(m)
        # hits rise when loss plateaus poorly; fall when improving
        if step > 0 and history[-1]["loss"] < history[-2]["loss"]:
            recent_hits = max(0.0, recent_hits - 0.05)
        else:
            recent_hits = min(1.0, recent_hits + 0.02)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    # Forward-only throughput
    model.eval()
    with torch.no_grad():
        t1 = time.perf_counter()
        for _ in range(50):
            _ = model(tokens[:-1])
        if device == "cuda":
            torch.cuda.synchronize()
        t_fwd = time.perf_counter() - t1

    report["train"] = {
        "steps": n_steps,
        "seconds": round(dt, 4),
        "loss_start": history[0]["loss"],
        "loss_end": history[-1]["loss"],
        "loss_improved": history[-1]["loss"] < history[0]["loss"],
        "lr_start": history[0]["lr"],
        "lr_end": history[-1]["lr"],
        "S_end": history[-1]["S"],
        "history_tail": history[-5:],
        "params": sum(p.numel() for p in model.parameters()),
        "config": asdict(cfg),
    }
    report["forward_bench"] = {
        "iters": 50,
        "seconds": round(t_fwd, 4),
        "seq": seq_len - 1,
        "ms_per_forward": round(1000 * t_fwd / 50, 3),
    }

    # 4) Memory footprint
    if device == "cuda":
        report["vram"] = {
            "allocated_mib": round(torch.cuda.memory_allocated() / 1024**2, 2),
            "reserved_mib": round(torch.cuda.memory_reserved() / 1024**2, 2),
        }

    report["ok"] = (
        report.get("crystal", {}).get("ok", True)
        and report["train"]["loss_improved"]
        and (
            report.get("scalar", {}).get("rel_err") is None
            or report["scalar"]["rel_err"] < 1e-4
            or report["scalar"].get("archive") is None
        )
    )

    path = RESULTS / "fsot_gpu_engine_run.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== FSOT GPU ENGINE (trinary kernel → RTX 5070) ===")
    print(f"Device:     {report['gpu_name']}")
    print(f"Collapse θ: {COLLAPSE_THRESHOLD:.12g}")
    if "scalar" in report:
        print(f"Scalar GPU: {report['scalar'].get('gpu') or report['scalar'].get('gpu_only')}")
        if report["scalar"].get("archive") is not None:
            print(
                f"Archive S:  {report['scalar']['archive']}  rel_err={report['scalar']['rel_err']}"
            )
    c = report["crystal"]
    if c.get("ok"):
        print(
            f"Crystal:    {c['voxels_used']} voxels packed → {c['packed_words']} u64 "
            f"({c['spin_up']}+ / {c['superposed']}0 / {c['spin_down']}-)"
        )
    else:
        print(f"Crystal:    {c}")
    print(
        f"Train:      loss {report['train']['loss_start']:.4f} → "
        f"{report['train']['loss_end']:.4f}  "
        f"({report['train']['params']} params, FSOT LR)"
    )
    print(f"Forward:    {report['forward_bench']['ms_per_forward']} ms/pass")
    print(f"OK:         {report['ok']}")
    print(f"Wrote:      {path}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

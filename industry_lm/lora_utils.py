"""Minimal LoRA wrap for nn.Linear (no peft required for custom FSOT attn)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(type(base))
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f = base.in_features
        out_f = base.out_features
        self.r = r
        self.scaling = alpha / r
        dev = base.weight.device
        dt = base.weight.dtype
        # A: r x in, B: out x r  (y += x @ A.T @ B.T * scale)
        self.lora_A = nn.Parameter(torch.zeros(r, in_f, device=dev, dtype=dt))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.base_bias = base.bias is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.base.weight, self.base.bias)
        # low-rank update
        lora = (x @ self.lora_A.t()) @ self.lora_B.t()
        return result + lora * self.scaling


def inject_lora_into_fsot_attn(model: nn.Module, r: int = 16, alpha: float = 32.0) -> int:
    """Wrap q/k/v/o projs on every FsotLlamaAttention. Returns # modules wrapped."""
    n = 0
    for layer in model.model.layers:
        attn = layer.self_attn
        name = type(attn).__name__
        if name != "FsotLlamaAttention":
            continue
        for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(attn, attr)
            if isinstance(lin, LoRALinear):
                continue
            setattr(attn, attr, LoRALinear(lin, r=r, alpha=alpha))
            n += 1
    return n


def lora_parameters(model: nn.Module):
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            yield p


def freeze_non_lora(model: nn.Module):
    for n, p in model.named_parameters():
        p.requires_grad_("lora_A" in n or "lora_B" in n)

"""
FSOT-gated industry attention (quality recovery path).

FSOT law still decides *who speaks*:
  collapse θ + coherence gate → key mask
Industry math among active keys:
  scaled dot-product + softmax (only on allowed keys)

This is the hybrid that keeps SafeTensors training signal while
using FSOT sparsity — first step toward beating baseline quality+speed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
from fsot_lib.seeds import COLLAPSE_THRESHOLD  # noqa: E402

COH_GATE = 0.5


def fsot_key_mask(k: torch.Tensor) -> torch.Tensor:
    """
    k: [B, H, S, D] → bool mask [B, H, S] True = active (coh > gate)
    """
    coh = (k.abs() > COLLAPSE_THRESHOLD).float().mean(dim=-1)
    return coh > COH_GATE


class FsotGatedLlamaAttention(nn.Module):
    """LlamaAttention with FSOT active-key mask on SDPA."""

    def __init__(self, src: LlamaAttention):
        super().__init__()
        self.config = src.config
        self.layer_idx = src.layer_idx
        self.head_dim = src.head_dim
        self.num_key_value_groups = src.num_key_value_groups
        self.scaling = src.scaling
        self.attention_dropout = src.attention_dropout
        self.is_causal = src.is_causal
        self.q_proj = src.q_proj
        self.k_proj = src.k_proj
        self.v_proj = src.v_proj
        self.o_proj = src.o_proj

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        B, H, S, D = query_states.shape
        # FSOT mask: inactive keys get -inf
        active = fsot_key_mask(key_states)  # B,H,S
        # If a head has zero active keys, fall back to all-active (stability)
        none = ~active.any(dim=-1, keepdim=True)  # B,H,1
        active = active | none

        # causal + FSOT key mask
        # attn_mask for SDPA: [B,H,S,S] True means keep (depending on API)
        # Use additive float mask
        neg = torch.finfo(query_states.dtype).min
        # start from zeros
        add_mask = torch.zeros(
            B, H, S, S, device=query_states.device, dtype=query_states.dtype
        )
        # causal: j > i → block
        causal = torch.triu(
            torch.ones(S, S, device=query_states.device, dtype=torch.bool), diagonal=1
        )
        add_mask = add_mask.masked_fill(causal.view(1, 1, S, S), neg)
        # inactive keys: for all queries, block key j if not active
        # active: B,H,S → broadcast to B,H,S,S on key dim
        key_block = ~active.unsqueeze(2)  # B,H,1,S
        add_mask = add_mask.masked_fill(key_block, neg)

        if attention_mask is not None:
            # merge if 4d
            if attention_mask.dim() == 4:
                add_mask = add_mask + attention_mask.to(add_mask.dtype)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=add_mask,
            dropout_p=0.0,
            is_causal=False,  # causal already in add_mask
            scale=self.scaling,
        )

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


def swap_all_gated(model: nn.Module) -> nn.Module:
    for i in range(len(model.model.layers)):
        src = model.model.layers[i].self_attn
        model.model.layers[i].self_attn = FsotGatedLlamaAttention(src)
    return model

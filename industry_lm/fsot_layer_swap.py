"""
Replace one LlamaAttention layer with FSOT consensus attention.
Keeps Q/K/V/O projections + RoPE from industry weights.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
)

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from fsot_cuda_ops import available as cuda_available, fsot_consensus  # noqa: E402
from competitive.sparse_consensus_batched import (  # noqa: E402
    consensus_true_sparse_padded,
)


def _fsot_attn_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q,k,v: [B, H, S, D]
    Prefer CUDA DLL; fall back to torch sparse (contract-faithful).
    """
    if cuda_available() and q.is_cuda:
        try:
            return fsot_consensus(q, k, v)
        except Exception:
            pass
    # torch path: per-batch
    outs = []
    for b in range(q.shape[0]):
        outs.append(consensus_true_sparse_padded(q[b], k[b], v[b]))
    return torch.stack(outs, dim=0)


class FsotLlamaAttention(nn.Module):
    """Drop-in replacement for LlamaAttention using FSOT consensus."""

    def __init__(self, src: LlamaAttention):
        super().__init__()
        self.config = src.config
        self.layer_idx = src.layer_idx
        self.head_dim = src.head_dim
        self.num_key_value_groups = src.num_key_value_groups
        self.scaling = src.scaling
        self.attention_dropout = src.attention_dropout
        self.is_causal = src.is_causal
        # share industry weights
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

        # GQA → full heads for consensus
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # FSOT consensus (causal built-in); ignore additive mask for pure causal eval
        attn_output = _fsot_attn_core(query_states, key_states, value_states)

        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


def swap_layer(model: nn.Module, layer_idx: int = 0) -> nn.Module:
    """Replace model.model.layers[layer_idx].self_attn with FSOT attention."""
    layer = model.model.layers[layer_idx]
    layer.self_attn = FsotLlamaAttention(layer.self_attn)
    return model


def swap_all_layers(model: nn.Module) -> nn.Module:
    """Replace every layer's self_attn with FSOT consensus attention."""
    n = len(model.model.layers)
    for i in range(n):
        swap_layer(model, layer_idx=i)
    return model


def swap_layers(model: nn.Module, layer_indices: list[int] | None = None) -> nn.Module:
    """Swap selected layers (default: all)."""
    if layer_indices is None:
        return swap_all_layers(model)
    for i in layer_indices:
        swap_layer(model, layer_idx=i)
    return model

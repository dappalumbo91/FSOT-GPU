#!/usr/bin/env python3
"""
Export a portable, code-agnostic schema for the industry model.

Any language can:
  1) Read portable_schema.json
  2) mmap the safetensors file by tensor name
  3) Implement the graph ops listed in graph_ir

FSOT backends replace specific op kinds (rms_norm, attn) without changing weights.
"""
from __future__ import annotations

import json
from pathlib import Path

from load_bank import load_bank, DEFAULT_MODEL

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "portable"


def build_graph_ir(config: dict) -> dict:
    """Minimal transformer IR (Llama/SmolLM style)."""
    n_layers = int(config.get("num_hidden_layers", 0))
    layers = []
    for i in range(n_layers):
        layers.append(
            {
                "id": f"layer_{i}",
                "ops": [
                    {"kind": "rms_norm", "weights": [f"model.layers.{i}.input_layernorm.weight"]},
                    {
                        "kind": "attn",
                        "variant": "industry_sdpa",  # replaceable with fsot_consensus
                        "weights": [
                            f"model.layers.{i}.self_attn.q_proj.weight",
                            f"model.layers.{i}.self_attn.k_proj.weight",
                            f"model.layers.{i}.self_attn.v_proj.weight",
                            f"model.layers.{i}.self_attn.o_proj.weight",
                        ],
                    },
                    {"kind": "residual_add"},
                    {
                        "kind": "rms_norm",
                        "weights": [f"model.layers.{i}.post_attention_layernorm.weight"],
                    },
                    {
                        "kind": "mlp_silu_gate",
                        "weights": [
                            f"model.layers.{i}.mlp.gate_proj.weight",
                            f"model.layers.{i}.mlp.up_proj.weight",
                            f"model.layers.{i}.mlp.down_proj.weight",
                        ],
                    },
                    {"kind": "residual_add"},
                ],
            }
        )
    return {
        "format": "fsot_portable_graph_v0",
        "architecture": config.get("model_type") or config.get("architectures"),
        "dims": {
            "hidden_size": config.get("hidden_size"),
            "intermediate_size": config.get("intermediate_size"),
            "num_attention_heads": config.get("num_attention_heads"),
            "num_key_value_heads": config.get("num_key_value_heads"),
            "num_hidden_layers": n_layers,
            "vocab_size": config.get("vocab_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "rms_norm_eps": config.get("rms_norm_eps"),
            "rope_theta": config.get("rope_theta"),
        },
        "entry": [
            {"kind": "embed", "weights": ["model.embed_tokens.weight"]},
        ],
        "layers": layers,
        "exit": [
            {"kind": "rms_norm", "weights": ["model.norm.weight"]},
            {"kind": "lm_head", "weights": ["lm_head.weight"], "note": "may tie embeddings"},
        ],
        "fsot_replaceable": {
            "attn": ["industry_sdpa", "fsot_consensus_cuda", "fsot_consensus_torch"],
            "rms_norm": ["industry_rms", "fsot_coherence_norm"],
        },
    }


def export(model_dir: Path | None = None) -> Path:
    bank = load_bank(model_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    schema = {
        "format": "fsot_portable_weights_v0",
        "purpose": "code-agnostic LLM weight bank for multi-language GPU hosts",
        "model": bank.summary(),
        "tensors": [t.__dict__ for t in bank.tensors],
        "weight_file": bank.safetensors_path,
        "graph_ir": build_graph_ir(bank.config),
        "backends": {
            "torch_hf": "industry_lm/baseline_hf.py",
            "fsot_bridge": "industry_lm/fsot_bridge.py",
            "future_rust": "read safetensors + graph_ir",
            "future_zig": "read safetensors + graph_ir",
            "cuda": "phase2_native_gpu/cuda (attn/norm kernels)",
        },
        "not_included": "FSOT-2.1-Instruct project (separate)",
    }
    out = OUT / "portable_schema.json"
    out.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    s = bank.summary()
    (OUT / "MANIFEST.md").write_text(
        "\n".join(
            [
                "# Portable bank",
                "",
                f"- model: {s['model_id']}",
                f"- params: {s['total_params']:,}",
                f"- bytes: {s['total_bytes']:,}",
                f"- tensors: {s['n_tensors']}",
                f"- safetensors: `{bank.safetensors_path}`",
                "- schema: `portable_schema.json`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out


if __name__ == "__main__":
    p = export()
    print("wrote", p)

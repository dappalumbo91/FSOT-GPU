#!/usr/bin/env python3
"""Load industry SafeTensors into a language-neutral weight bank."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "SmolLM2-135M-Instruct"


@dataclass
class TensorRef:
    name: str
    shape: list[int]
    dtype: str
    nbytes: int


@dataclass
class WeightBank:
    model_id: str
    model_dir: str
    safetensors_path: str
    config: dict[str, Any]
    tensors: list[TensorRef]

    def summary(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_dir": self.model_dir,
            "safetensors_path": self.safetensors_path,
            "n_tensors": len(self.tensors),
            "total_params": sum(int(__import__("math").prod(t.shape)) for t in self.tensors),
            "total_bytes": sum(t.nbytes for t in self.tensors),
            "config_keys": list(self.config.keys())[:20],
            "arch": self.config.get("architectures") or self.config.get("model_type"),
            "hidden_size": self.config.get("hidden_size"),
            "num_hidden_layers": self.config.get("num_hidden_layers"),
            "num_attention_heads": self.config.get("num_attention_heads"),
            "vocab_size": self.config.get("vocab_size"),
        }


def load_bank(model_dir: Path | None = None) -> WeightBank:
    model_dir = Path(model_dir or DEFAULT_MODEL)
    cfg_path = model_dir / "config.json"
    st_path = model_dir / "model.safetensors"
    if not st_path.is_file():
        # multi-shard
        shards = sorted(model_dir.glob("model-*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"No safetensors in {model_dir}")
        st_path = shards[0]

    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    tensors: list[TensorRef] = []
    with safe_open(str(st_path), framework="pt", device="cpu") as f:
        for name in f.keys():
            t = f.get_tensor(name)
            tensors.append(
                TensorRef(
                    name=name,
                    shape=list(t.shape),
                    dtype=str(t.dtype).replace("torch.", ""),
                    nbytes=int(t.numel() * t.element_size()),
                )
            )
    return WeightBank(
        model_id=config.get("_name_or_path")
        or model_dir.name
        or "unknown",
        model_dir=str(model_dir.resolve()),
        safetensors_path=str(st_path.resolve()),
        config=config,
        tensors=tensors,
    )


if __name__ == "__main__":
    b = load_bank()
    print(json.dumps(b.summary(), indent=2))

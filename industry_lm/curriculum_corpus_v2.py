#!/usr/bin/env python3
"""
Deeper curriculum corpus (Phase 1b): more 2.1 docs + multi-domain solidification.

Writes: results/industry_lm/curriculum_v2_chunks.jsonl
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config" / "curriculum_roots.json").read_text(encoding="utf-8"))
OUT = ROOT / "results" / "industry_lm"
OUT.mkdir(parents=True, exist_ok=True)

CHUNK = 850
OVERLAP = 100
MAX_CHUNKS = 20000
MAX_ARXIV = 40 * 1024 * 1024
PUBLIC_DOMAINS = [
    "nist_codata",
    "nasa_exoplanet",
    "space_weather",
    "pubchem",
    "rcsb_pdb",
    "openalex",
    "noaa_tides",
    "gbif",
    "cern_opendata",
    "uniprot",
    "world_bank",
    "consciousness",
    "anomaly_observables",
    "trinary_os",
]


def clean(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def chunk_text(text: str, source: str, kind: str, limit_chunks: int = 8000) -> list[dict]:
    text = clean(text)
    if len(text) < 40:
        return []
    rows = []
    i = n = 0
    while i < len(text) and n < limit_chunks:
        piece = text[i : i + CHUNK].strip()
        if len(piece) < 40:
            break
        rows.append({"text": piece, "source": source, "kind": kind})
        n += 1
        i += CHUNK - OVERLAP
    return rows


def load_lean(lean: Path) -> list[dict]:
    docs = lean / "docs"
    rows = []
    if not docs.is_dir():
        return rows
    for p in sorted(docs.rglob("*")):
        if p.suffix.lower() not in {".md", ".txt", ".yaml", ".yml", ".html"}:
            continue
        raw = p.read_text(encoding="utf-8", errors="ignore")
        if len(raw) > 2_000_000:
            raw = raw[:2_000_000]
        rows.extend(chunk_text(raw, str(p), "fsot_2_1_doc"))
    # also FSOT/ and verification text if present
    for sub in ("FSOT", "verification", "explorations"):
        d = lean / sub
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*.md"))[:80]:
            try:
                raw = p.read_text(encoding="utf-8", errors="ignore")[:500_000]
            except OSError:
                continue
            rows.extend(chunk_text(raw, str(p), "fsot_2_1_tree", limit_chunks=200))
    return rows


def load_arxiv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows, buf, size = [], [], 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            size += len(line.encode("utf-8", errors="ignore"))
            if size > MAX_ARXIV:
                break
            if line.startswith("[CAT]") and buf:
                rows.extend(chunk_text("".join(buf), str(path), "arxiv_fsot_core", 50))
                buf = [line]
            else:
                buf.append(line)
        if buf:
            rows.extend(chunk_text("".join(buf), str(path), "arxiv_fsot_core", 50))
    return rows


def load_public(public: Path) -> list[dict]:
    rows = []
    for dom in PUBLIC_DOMAINS:
        d = public / dom
        if not d.exists():
            continue
        n_before = len(rows)
        files = []
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".json", ".txt", ".md", ".csv", ".tsv", ".yaml", ".yml"}:
                continue
            if p.stat().st_size > 3_000_000:
                continue
            files.append(p)
            if len(files) >= 40:
                break
        for p in files:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")[:400_000]
            except OSError:
                continue
            # json: flatten a bit
            if p.suffix.lower() == ".json":
                try:
                    obj = json.loads(text)
                    text = json.dumps(obj, indent=0)[:300_000]
                except json.JSONDecodeError:
                    pass
            rows.extend(
                chunk_text(text, str(p), f"solidification_{dom}", limit_chunks=80)
            )
        print(f"  domain {dom}: +{len(rows)-n_before} chunks")
    return rows


def primer() -> list[dict]:
    t = """
FSOT Fluid Spacetime Omni-Theory operator primer.
Collapse threshold theta = C_eff * P_var approximately 0.9174663774653723.
Trinary: up if x>theta, down if x<-theta, else neutral.
Coherence gate: active key when sharp/D > 0.5.
Consensus attention: no softmax exp; trit agreement weights in [-1,1]; causal; O(H S A D).
D_eff is dimensional calibration of the interaction regime for the FSOT scalar.
Suction-poof LR: suction*(1-poof*tanh(loss))*exp(-alpha*hits)*K.
Theory authority: FSOT-2.1-Lean multi-prover verification and physical archive ledgers.
FSOT-GPU hosts pure FSOT consensus on CUDA sm_120 for language model layers.
"""
    return chunk_text(t, "primer", "fsot_operator_primer")


def main():
    lean = Path(CFG["lean_hub"])
    train = Path(CFG["training_data"])
    public = Path(CFG["public_data"])
    chunks = []
    chunks.extend(primer())
    lean_rows = load_lean(lean)
    print("lean", len(lean_rows))
    chunks.extend(lean_rows)
    arx = load_arxiv(train / "arxiv_fsot_core.txt")
    print("arxiv", len(arx))
    chunks.extend(arx)
    pub = load_public(public)
    print("public", len(pub))
    chunks.extend(pub)

    if len(chunks) > MAX_CHUNKS:
        # prioritize primer+lean+public over arxiv bulk
        keep = []
        rest = []
        for c in chunks:
            if c["kind"] in (
                "fsot_operator_primer",
                "fsot_2_1_doc",
                "fsot_2_1_tree",
            ) or c["kind"].startswith("solidification_"):
                keep.append(c)
            else:
                rest.append(c)
        need = max(MAX_CHUNKS - len(keep), 0)
        step = max(len(rest) // max(need, 1), 1)
        chunks = keep + rest[::step][:need]

    outp = OUT / "curriculum_v2_chunks.jsonl"
    with outp.open("w", encoding="utf-8") as f:
        for r in chunks:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by = {}
    for r in chunks:
        by[r["kind"]] = by.get(r["kind"], 0) + 1
    meta = {"n": len(chunks), "by_kind": by, "out": str(outp)}
    (OUT / "curriculum_v2_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", outp, meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

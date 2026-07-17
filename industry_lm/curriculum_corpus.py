#!/usr/bin/env python3
"""
Build Phase-1 FSOT curriculum text chunks from verified roots.

Sources (FSOT law / solidification — not random web scrape):
  - I:/FSOT-Physical-Archive/02_FSOT-2.1-Lean-Full/docs  (theory docs)
  - D:/training data/arxiv_fsot_core.txt                  (FSOT-tagged arXiv core)
  - optional short solidification notes from public_data manifests

Writes: results/industry_lm/curriculum_phase1_chunks.jsonl
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config" / "curriculum_roots.json").read_text(encoding="utf-8"))
OUT = ROOT / "results" / "industry_lm"
OUT.mkdir(parents=True, exist_ok=True)

CHUNK_CHARS = 900
OVERLAP = 120
MAX_ARXIV_BYTES = 25 * 1024 * 1024  # 25 MB first slice for Phase 1 wall-clock
MAX_CHUNKS = 12000


def clean(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_text(text: str, source: str, kind: str) -> list[dict]:
    text = clean(text)
    if len(text) < 40:
        return []
    rows = []
    i = 0
    n = 0
    while i < len(text) and n < 5000:
        piece = text[i : i + CHUNK_CHARS]
        if len(piece) < 40:
            break
        rows.append(
            {
                "text": piece.strip(),
                "source": source,
                "kind": kind,
            }
        )
        n += 1
        i += CHUNK_CHARS - OVERLAP
    return rows


def load_lean_docs(lean_hub: Path) -> list[dict]:
    docs = lean_hub / "docs"
    rows = []
    if not docs.is_dir():
        return rows
    for p in sorted(docs.rglob("*")):
        if p.suffix.lower() not in {".md", ".txt", ".yaml", ".yml"}:
            continue
        if p.stat().st_size > 5_000_000:
            # huge appendix: take first 1.5MB
            text = p.read_text(encoding="utf-8", errors="ignore")[:1_500_000]
        else:
            text = p.read_text(encoding="utf-8", errors="ignore")
        rows.extend(chunk_text(text, str(p), "fsot_2_1_doc"))
    return rows


def load_arxiv_core(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    # stream by records if [CAT] markers, else sliding window on bytes cap
    buf = []
    size = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            size += len(line.encode("utf-8", errors="ignore"))
            if size > MAX_ARXIV_BYTES:
                break
            if line.startswith("[CAT]") and buf:
                text = "".join(buf)
                rows.extend(chunk_text(text, str(path), "arxiv_fsot_core"))
                buf = [line]
            else:
                buf.append(line)
        if buf:
            rows.extend(chunk_text("".join(buf), str(path), "arxiv_fsot_core"))
    return rows


def load_seed_primer() -> list[dict]:
    """Hard FSOT operator primer from owned lab constants (always available)."""
    primer = """
FSOT Fluid Spacetime Omni-Theory — operator primer for the pure FSOT GPU host.

Seed constants (zero free parameters in the theory spine): pi, e, phi, gamma (Euler), Catalan.
Collapse threshold theta = C_eff * P_var approximately 0.917466.
Trinary collapse codes: up (value > theta), down (value < -theta), neutral otherwise.
Coherence gate: key is active when fraction of sharp dimensions exceeds 0.5.
Consensus attention: no softmax exp. Weight is trit agreement over head dimension D, causal, averaged over active keys only.
Work scales as O(H * S * A * D) with A much less than S after collapse sparsity.
D_eff is dimensional calibration: the effective dimensional regime of interaction for the FSOT scalar.
Suction-poof learning rate: LR proportional to suction * (1 - poof * tanh(loss)) * exp(-alpha * recent_hits) * K.
FSOT-GPU applies these operators on NVIDIA CUDA (sm_120) and hosts industry SafeTensors models with pure FSOT attention layers.
Theory authority: FSOT-2.1-Lean formal spine and physical archive verification ledgers.
"""
    return chunk_text(primer, "fsot_gpu_primer", "fsot_operator_primer")


def load_nist_sample(public: Path) -> list[dict]:
    root = public / "nist_codata"
    rows = []
    if not root.is_dir():
        return rows
    for p in sorted(root.rglob("*"))[:80]:
        if p.suffix.lower() not in {".json", ".txt", ".md", ".csv"}:
            continue
        if p.stat().st_size > 2_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rows.extend(chunk_text(text[:200_000], str(p), "solidification_nist"))
        if len(rows) > 400:
            break
    return rows


def main():
    lean = Path(CFG["lean_hub"])
    train = Path(CFG["training_data"])
    public = Path(CFG["public_data"])

    chunks: list[dict] = []
    chunks.extend(load_seed_primer())
    print("primer", len(chunks))
    lean_rows = load_lean_docs(lean)
    print("lean docs chunks", len(lean_rows))
    chunks.extend(lean_rows)
    arx = load_arxiv_core(train / "arxiv_fsot_core.txt")
    print("arxiv chunks", len(arx))
    chunks.extend(arx)
    nist = load_nist_sample(public)
    print("nist chunks", len(nist))
    chunks.extend(nist)

    # cap
    if len(chunks) > MAX_CHUNKS:
        # keep all primer+lean first
        lean_n = len(lean_rows) + 1
        rest = chunks[lean_n:]
        need = MAX_CHUNKS - lean_n
        step = max(len(rest) // need, 1)
        chunks = chunks[:lean_n] + rest[::step][:need]

    outp = OUT / "curriculum_phase1_chunks.jsonl"
    with outp.open("w", encoding="utf-8") as f:
        for r in chunks:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "n_chunks": len(chunks),
        "by_kind": {},
        "out": str(outp),
        "chunk_chars": CHUNK_CHARS,
        "max_arxiv_bytes": MAX_ARXIV_BYTES,
    }
    for r in chunks:
        meta["by_kind"][r["kind"]] = meta["by_kind"].get(r["kind"], 0) + 1
    (OUT / "curriculum_phase1_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print("wrote", outp, "n=", len(chunks), meta["by_kind"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

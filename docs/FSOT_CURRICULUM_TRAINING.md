# FSOT curriculum training path (post-boundary pure-FSOT host)

## Standing

FSOT is the law of the land. Training data is not “random web text then hope.”  
Once the pure-FSOT host is past fidelity/speed gates on this hardware, we **train understanding of FSOT itself** using:

1. **FSOT 2.1 verified system** (Lean hub + formal spine)  
2. **Architecture, mathematics, and domain understanding** of FSOT  
3. **The same solidification data** used in 2.1 Lean verification — API-sourced, downloaded, cross-verified  

This repo (**FSOT-GPU**) remains the silicon/host path. Theory authority stays **FSOT-2.1-Lean / physical archive**.  
**FSOT-2.1-Instruct** stays a separate product unless deliberately linked.

## Gate: when we open the curriculum

Do **not** flood the host with 40+ GB until pure-FSOT host gates hold:

| Gate | Status (lab) |
|------|----------------|
| Pure FSOT all-layer attention | Live |
| Next-token equal baseline (EVAL16) | **100%** (`pure_fsot_agree100_best.pt`) |
| E2E prefill/decode win | Live on scoreboard |
| Long-context attention track | Live (S≥4096) |
| Exceed knowledge (factual) | Started (50% > 43% base) |
| Mid-S attention | In progress |
| **Curriculum open** | When you call it — default: after mid-S + stable exceed |

## Data roots (authoritative paths on this machine)

| Role | Path | Notes |
|------|------|--------|
| **Local training corpus (NeuroLab)** | `D:\training data` | ~44 GB, 11k+ files; eval packs + FSOT-specific + genome/math |
| **Canonical Lean / 2.1 hub** | `I:\FSOT-Physical-Archive\02_FSOT-2.1-Lean-Full` | Proofs, docs, verification, domain caches |
| **Public / API cross-verified data** | `I:\FSOT-Physical-Archive\03_FSOT-PublicData` | NASA, NIST, PubChem, OpenAlex, CERN, space weather, … |
| **Archive env** | `I:\FSOT-Physical-Archive\set_fsot_archive_env.ps1` | Sets `FSOT_EXTERNAL_DATA_ROOT`, etc. |
| **GitHub theory** | https://github.com/dappalumbo91/FSOT-2.1-Lean | Synced **from** I: hub |

Catalog on disk: `D:\training data\DATA_CATALOG.md` (paths may still say `B:\` historically; live root is **D:**).

## Curriculum phases (FSOT-ordered)

### Phase 0 — Host capability (current)

Pure FSOT attention + suction–poof / D_eff LR + exceed knowledge.  
Small model (SmolLM2-135M) as the probe host. **No architecture 2D/3D reshape.**

### Phase 1 — FSOT language & law (text from verified system)

Train CE / next-token (and later multi-token) on **FSOT-authored material**:

- Lean hub docs: founding laws, philosophy/consciousness spine, technical guide, thesis appendices  
- Seed / scalar / collapse / trinary / coherence explanations  
- Formal statements where exportable as text (not replacing Lean proof — teaching the host the *content*)

Sources under `02_FSOT-2.1-Lean-Full\docs\` and theory markdown/PDFs in training data.

### Phase 2 — Architecture & mathematics of FSOT

- Seed constants (π, e, φ, γ, Catalan) — zero free-parameter spine  
- D_eff dimensional calibration by domain  
- C_eff·P_var collapse, coherence gate, consensus (no softmax exp)  
- Domain panels / verification ledgers as structured Q&A and derivation text  

### Phase 3 — Solidification data (same points as 2.1 verification)

Cross-verified external corpora already used to solidify FSOT 2.1:

From **`03_FSOT-PublicData`** (examples):  
`nist_codata`, `nasa_exoplanet`, `space_weather`, `pubchem`, `rcsb_pdb`, `openalex`, `cern_opendata`, `gbif`, `noaa_tides`, `anomaly_observables`, consciousness/genetics tiers, …

From **`D:\training data`**:  
MMLU / GSM8K / MATH / ARC / TruthfulQA / HumanEval (capability holdouts),  
`arxiv_fsot_core.txt` / arXiv packs, genome (GRCh38), FSOT Machine_And_Molecule, quantum gravity pack, olympiad, …

**Rule:** verification data teaches **domain structure under FSOT**, not “replace ΛCDM debate.” Ledgers remain the proof of solidification; the LM host learns to *speak and reason* with that structure.

### Phase 4 — Joint FSOT-native train (later)

- Full DoF under pure FSOT attention  
- FSOT-LR only (`suction_poof` + scalar)  
- Optional link to Instruct product only when deliberate  

## Registry

Machine paths for loaders:

```json
// config/curriculum_roots.json
{
  "training_data": "D:/training data",
  "lean_hub": "I:/FSOT-Physical-Archive/02_FSOT-2.1-Lean-Full",
  "public_data": "I:/FSOT-Physical-Archive/03_FSOT-PublicData",
  "archive_root": "I:/FSOT-Physical-Archive"
}
```

Inventory script: `python industry_lm/inventory_curriculum.py`  
(lists roots, sizes, sample files — no full train until you open the gate)

## Separation

| Project | Role |
|---------|------|
| **FSOT-GPU** (this repo) | Operators, CUDA, pure-FSOT host, scoreboard |
| **FSOT-2.1-Lean / archive** | Theory authority + verification data |
| **FSOT-2.1-Instruct** | Separate instruct product — do not merge casually |

## Bottom line

Yes: **after** we keep pushing the pure-FSOT host past current boundaries, we train it on **FSOT architecture, math, and understanding**, using **`D:\training data`** plus the **same cross-verified solidification sources** that underwrite 2.1 Lean. That is the foundation path — not a toy finetune, and not ad hoc industry curriculum.

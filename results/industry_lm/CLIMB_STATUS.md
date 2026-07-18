# Open-source SOTA climb status (honest ledger)

**Goal:** past 62% fact tie + dominate same-class open hosts.

## 1. Deeper curriculum — DONE (corpus)

- `curriculum_v2_chunks.jsonl`: **20,000** chunks  
- 2.1 Lean docs + tree markdown  
- Solidification domains: NIST, NASA exoplanet, PubChem, OpenAlex, GBIF, CERN, UniProt, World Bank, consciousness, anomaly_observables, trinary_os, …  
- arxiv_fsot_core (sampled)  

## 2. Public capability packs — SMOKED

| Ckpt | ARC-Easy | GSM8K | MATH | Macro |
|------|----------|-------|------|-------|
| Industry baseline | 12% | 68% | 33% | 38% |
| pure_fsot_sota_climb_best | **20%** (+8) | 15% | 3% | 13% |
| pure_fsot_curriculum_best | 12% | 0% | 0% | 4% |

**Win:** ARC-Easy under climb ckpt **beats baseline**.  
**Gap:** free-gen GSM8K/MATH still industry-led (numeric generation collapse under heavy fact CE).

## 3. Mid-S attention — PARTIAL

| S | FSOT vs SDPA |
|---|--------------|
| 64–1024 | still lose |
| **2048** | **WIN ~1.09×** |
| 4096 | ~tie / slight lose (variance) |
| **8192** | **WIN ~1.75×** |

Long-context domain solid. Mid-short still fused-SDPA sweet spot.

## 4. Fact + cleaner FSOT gen — CLIMBING

| Metric | Start (curriculum) | Best climb | Baseline host |
|--------|--------------------|------------|---------------|
| Multi-token fact (14) | 29% | **64%** | 71% |
| FSOT literacy | 38% | **50%** peaks | 25% |
| Agree16 | 100% | 94–100% | 100% self |

**Past old 62% first-token tie metric:** multi-token board is harder; we **closed 29→64** toward 71 base, not yet exceed.  
**Literacy:** pure FSOT **beats** baseline host on FSOT Q&A.

## Next levers (still FSOT-only)

1. Arithmetic-specialized CE without mode-collapse (mask/stabilize numeric tokens).  
2. GSM8K-format training mixed lighter with curriculum (preserve ARC win).  
3. Mid-S: more collapse-aware packing / occupancy (S=256–1024).  
4. Prefer **curriculum_best** for literacy/host; **climb_best** only when fact metric improves without GSM death.

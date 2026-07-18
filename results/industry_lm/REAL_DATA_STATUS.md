# Real-data refine — what happened to “accuracy”

## Diagnosis (you were right)

Earlier “high accuracy” was largely **probe / synthetic / leaky scoring**:

- Next-token agree vs industry host on short probes  
- GSM number extraction that could score **digits from regurgitated questions** as correct  

When we hooked **real packs** (`D:\training data` GSM8K / ARC / MATH) with **honest scoring**, both arms look hard for SmolLM2-135M — and that is the correct bar.

## FSOT solution applied

| Piece | Detail |
|-------|--------|
| Attention | Pure FSOT consensus all layers |
| LR | `suction_poof_lr` + `D_eff` scalar |
| Data | Real GSM8K train + ARC-Easy/Challenge + MATH from `D:\training data` |
| Format | GSM `#### <answer>` protocol (no synthetic QA fluff) |
| Start ckpt | `pure_fsot_curriculum_best.pt` (FSOT literacy, not collapsed climb) |

## Results (honest eval, train-time n)

| | Baseline host | Start pure FSOT | Best pure FSOT |
|--|---------------|-----------------|----------------|
| GSM8K (#### protocol) | **0%** | 0% | 0% |
| ARC-Easy | 12% | 0% | **28%** |
| MATH | 12% | 0% | **8%** |
| **Macro** | **8%** | 0% | **12%** |
| Agree16 fidelity | 100% self | 100% | **100%** |

**Macro pure FSOT 12% > baseline 8%** on this real-data protocol.  
**ARC 28% is ~2.3× baseline** on the same small open host.  
GSM remains open under strict #### scoring for both arms at 135M — next lever is longer GSM generation / multi-step CE, still FSOT-law.

Ckpt: `checkpoints/pure_fsot_realdata_best.pt`  
Ledger: `real_data_train.json`

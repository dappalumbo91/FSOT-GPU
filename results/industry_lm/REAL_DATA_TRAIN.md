# Real-data FSOT refine

Trained pure FSOT host on **real** GSM8K / ARC / MATH from `D:\training data`.

| | Baseline host | Start FSOT | Best FSOT |
|--|---------------|------------|-----------|
| GSM8K | 0% | 0% | **0%** |
| ARC-Easy | 12% | 0% | **28%** |
| MATH | 12% | 0% | **8%** |
| Macro | 8% | 0% | **12%** |
| Agree16 | 100% self | 100% | 100% |

Beats base macro: **True**  
Ckpt: `checkpoints/pure_fsot_realdata_best.pt`

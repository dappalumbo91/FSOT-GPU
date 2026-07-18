# FSOT 12×3 fast train

**Protocol:** 12 epochs × 3 real packs (GSM8K, ARC, MATH)  
**LR:** FSOT-derived only (`derive_fsot_lr_plan` + `fsot_epoch_lr`)

| | Baseline | Start | Best |
|--|----------|-------|------|
| GSM | 0% | 0% | **0%** |
| ARC | 12% | 30% | **35%** |
| MATH | 8% | 4% | **0%** |
| Macro | 7% | 11% | **12%** |
| Agree | 100% | 100% | 100% |

lr0=2.053e-05 floor=3.151e-06 ceil=3.000e-05  
Beats base macro: **True**  
Ckpt: `pure_fsot_12x3_best.pt`

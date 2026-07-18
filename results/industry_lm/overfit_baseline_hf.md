# Overfit metrics — overfit_baseline_hf

**gen_score:** 0.114  
**mean hold acc:** 16.1%  
**mean train acc:** 20.8%  
**mean overfit gap (train−hold):** +4.7%  
**max gap:** +16.7%  
**overfit_flag:** True (threshold 8%)  
**note:** overfit_signature

| Split | Train acc | Hold acc | Gap (train−hold) |
|-------|-----------|----------|------------------|
| arc_easy | 25.0% | 8.3% | +16.7% |
| arc_challenge | 12.5% | 12.5% | +0.0% |
| gsm_first_digit | 25.0% | 27.5% | -2.5% |

## How to read this

- **Gap ↑ while train ↑, hold flat/↓** → overfitting direction — reject step.
- **Hold ↑ and gap flat/↓** → generalization direction — accept.
- **gen_score** is what the system optimizes: hold quality minus gap penalty.

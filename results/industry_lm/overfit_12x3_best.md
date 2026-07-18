# Overfit metrics — overfit_12x3_best

**gen_score:** 0.272  
**mean hold acc:** 30.3%  
**mean train acc:** 33.3%  
**mean overfit gap (train−hold):** +3.1%  
**max gap:** +16.7%  
**overfit_flag:** True (threshold 8%)  
**note:** overfit_signature

| Split | Train acc | Hold acc | Gap (train−hold) |
|-------|-----------|----------|------------------|
| arc_easy | 35.0% | 18.3% | +16.7% |
| arc_challenge | 40.0% | 42.5% | -2.5% |
| gsm_first_digit | 25.0% | 30.0% | -5.0% |

## How to read this

- **Gap ↑ while train ↑, hold flat/↓** → overfitting direction — reject step.
- **Hold ↑ and gap flat/↓** → generalization direction — accept.
- **gen_score** is what the system optimizes: hold quality minus gap penalty.

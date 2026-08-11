# Overfit metrics — overfit_capability_recovery

**gen_score:** 0.325  
**mean hold acc:** 32.5%  
**mean train acc:** 24.2%  
**mean overfit gap (train−hold):** -8.3%  
**max gap:** -5.0%  
**overfit_flag:** False (threshold 8%)  
**note:** gap_within_threshold

| Split | Train acc | Hold acc | Gap (train−hold) |
|-------|-----------|----------|------------------|
| arc_easy | 20.0% | 35.0% | -15.0% |
| arc_challenge | 27.5% | 32.5% | -5.0% |
| gsm_first_digit | 25.0% | 30.0% | -5.0% |

## How to read this

- **Gap ↑ while train ↑, hold flat/↓** → overfitting direction — reject step.
- **Hold ↑ and gap flat/↓** → generalization direction — accept.
- **gen_score** is what the system optimizes: hold quality minus gap penalty.

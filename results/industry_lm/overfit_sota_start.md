# Overfit metrics — overfit_sota_start

**gen_score:** 0.292  
**mean hold acc:** 29.2%  
**mean train acc:** 22.5%  
**mean overfit gap (train−hold):** -6.7%  
**max gap:** -5.0%  
**overfit_flag:** False (threshold 8%)  
**note:** gap_within_threshold

| Split | Train acc | Hold acc | Gap (train−hold) |
|-------|-----------|----------|------------------|
| arc_easy | 25.0% | 30.0% | -5.0% |
| arc_challenge | 17.5% | 27.5% | -10.0% |
| gsm_first_digit | 25.0% | 30.0% | -5.0% |

## How to read this

- **Gap ↑ while train ↑, hold flat/↓** → overfitting direction — reject step.
- **Hold ↑ and gap flat/↓** → generalization direction — accept.
- **gen_score** is what the system optimizes: hold quality minus gap penalty.

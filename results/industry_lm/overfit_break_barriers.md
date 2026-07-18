# Overfit metrics — overfit_break_barriers

**gen_score:** 0.319  
**mean hold acc:** 31.9%  
**mean train acc:** 21.7%  
**mean overfit gap (train−hold):** -10.3%  
**max gap:** -5.0%  
**overfit_flag:** False (threshold 8%)  
**note:** gap_within_threshold

| Split | Train acc | Hold acc | Gap (train−hold) |
|-------|-----------|----------|------------------|
| arc_easy | 25.0% | 33.3% | -8.3% |
| arc_challenge | 15.0% | 32.5% | -17.5% |
| gsm_first_digit | 25.0% | 30.0% | -5.0% |

## How to read this

- **Gap ↑ while train ↑, hold flat/↓** → overfitting direction — reject step.
- **Hold ↑ and gap flat/↓** → generalization direction — accept.
- **gen_score** is what the system optimizes: hold quality minus gap penalty.

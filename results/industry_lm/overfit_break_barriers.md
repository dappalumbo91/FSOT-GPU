# Overfit metrics — overfit_break_barriers

**gen_score:** 0.308  
**mean hold acc:** 30.8%  
**mean train acc:** 25.8%  
**mean overfit gap (train−hold):** -5.0%  
**max gap:** +2.5%  
**overfit_flag:** False (threshold 8%)  
**note:** gap_within_threshold

| Split | Train acc | Hold acc | Gap (train−hold) |
|-------|-----------|----------|------------------|
| arc_easy | 27.5% | 25.0% | +2.5% |
| arc_challenge | 25.0% | 37.5% | -12.5% |
| gsm_first_digit | 25.0% | 30.0% | -5.0% |

## How to read this

- **Gap ↑ while train ↑, hold flat/↓** → overfitting direction — reject step.
- **Hold ↑ and gap flat/↓** → generalization direction — accept.
- **gen_score** is what the system optimizes: hold quality minus gap penalty.

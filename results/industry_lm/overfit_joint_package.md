# Overfit metrics — overfit_joint_package

**gen_score:** 0.300  
**mean hold acc:** 30.0%  
**mean train acc:** 29.2%  
**mean overfit gap (train−hold):** -0.8%  
**max gap:** +0.0%  
**overfit_flag:** False (threshold 8%)  
**note:** gap_within_threshold

| Split | Train acc | Hold acc | Gap (train−hold) |
|-------|-----------|----------|------------------|
| arc_easy | 25.0% | 25.0% | +0.0% |
| arc_challenge | 37.5% | 37.5% | +0.0% |
| gsm_first_digit | 25.0% | 27.5% | -2.5% |

## How to read this

- **Gap ↑ while train ↑, hold flat/↓** → overfitting direction — reject step.
- **Hold ↑ and gap flat/↓** → generalization direction — accept.
- **gen_score** is what the system optimizes: hold quality minus gap penalty.

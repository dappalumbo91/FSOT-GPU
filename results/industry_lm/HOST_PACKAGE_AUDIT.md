# Host package audit (joint capability)

Single-axis climbs that regress other floors are **not** FSOT application.
Rank is **package score**: ARC + agree + gen + digit − mode-collapse penalty.

| Package | ARC min | Space dig | Mode | Agree | gen | Host |
|---------|---------|-----------|------|-------|-----|------|
| 1.729 | 22% | 28% | 1@22% | 100% | 0.261 | `pure_fsot_hardware_sota_best.pt` |
| 1.670 | 18% | 30% | 1@38% | 100% | 0.272 | `pure_fsot_sota_standard_best.pt` |
| 1.670 | 18% | 30% | 1@38% | 100% | 0.272 | `pure_fsot_data_driven_best.pt` |
| 1.670 | 18% | 30% | 1@38% | 100% | 0.272 | `pure_fsot_granular_best.pt` |
| 1.594 | 25% | 32% | 1@85% | 100% | 0.308 | `pure_fsot_barrier_lab_best.pt` |
| 1.499 | 25% | 30% | 1@100% | 100% | 0.289 | `pure_fsot_answer_locked_best.pt` |
| 1.460 | 22% | 28% | 1@95% | 100% | 0.314 | `pure_fsot_gsm_locked_best.pt` |
| 1.427 | 22% | 30% | 1@100% | 100% | 0.278 | `pure_fsot_arc_locked_best.pt` |
| 1.369 | 20% | 30% | 1@100% | 100% | 0.228 | `pure_fsot_realdata_best.pt` |
| 1.358 | 18% | 30% | 1@100% | 100% | 0.272 | `pure_fsot_12x3_best.pt` |
| 1.004 | 13% | 0% |  @98% | 100% | 0.089 | `pure_fsot_agree100_best.pt` |
| 0.908 | 0% | 10% | 0@52% | 94% | 0.017 | `pure_fsot_sota_climb_best.pt` |
| 0.813 | 0% | 30% | 1@100% | 100% | 0.017 | `pure_fsot_curriculum_best.pt` |

**Spine host (highest package):** `pure_fsot_hardware_sota_best.pt`

Rule: all future train must **start from spine** and **reject any step**
that drops any frozen floor (ARC, agree, gen, mode mass, space-digit).

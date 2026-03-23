# Experiment Matrix

## Purpose

This file defines the paper-track experiment structure for the current project.

The project should now be presented as a focused benchmark-and-method paper, not as a zoo of unrelated agent branches.

## Main Comparison Table

The primary comparison table should include:

1. `heuristic`
2. `hierarchical BC-only`
3. `hierarchical BC + DAgger`
4. `hierarchical BC + DAgger + RL`

Primary metrics:

- `win_rate`
- `tissue_damage`
- `host_utility`
- `episode_length`
- `recruited_neutrophils`

Secondary behavioral metrics:

- `attack_actions`
- `signal_actions`
- `blind_signal_actions`
- `stay_actions`
- `coverage_ratio`
- `visible_bacteria_ratio`

## Reproducibility Table

The reproducibility table should report:

- training seed `42`
- training seed `43`
- training seed `44`
- heuristic reference on the same 100-seed evaluation set
- learned mean across seeds

Core interpretation:

- the learned method is reproducibly competitive
- the best run beats the heuristic
- seed variance is real, so the correct claim is competitiveness plus tissue preservation, not universal dominance

## Robustness Table

The robustness table should compare `heuristic` vs frozen `BC + DAgger` on:

1. `baseline_reference`
2. `high_bacteria_stochasticity`
3. `hotspot_overload`
4. `recruitment_delay_stress`
5. `fragile_tissue`
6. `narrow_bottleneck`

The strongest robustness wins to emphasize in the paper narrative are:

- `high_bacteria_stochasticity`
- `hotspot_overload`
- `recruitment_delay_stress`

## What Is Not The Main Story

These should not be the headline contribution:

- runtime MCTS
- flat PPO
- unconstrained RL fine-tuning
- arbitrary extra training without held-out validation

## Next Packaging Step

The project is now at the stage of:

- frozen mainline reporting
- final table construction
- figure and demo preparation
- paper writing

Not:

- searching for a brand new control algorithm

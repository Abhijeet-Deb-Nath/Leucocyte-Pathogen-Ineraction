# Robustness Artifacts

This directory stores paper-track robustness evaluations for the frozen learned controller.

Current intended layout:

- `results/robustness/hier_dagger_main/`

Contents:

- aggregate robustness summary
- suite manifest
- per-suite summary CSVs
- optional per-suite detailed CSVs

Current interpretation:

- the frozen hierarchical `BC + DAgger` controller remains competitive or stronger than the heuristic in several adversarial stress settings
- `hotspot_overload`, `recruitment_delay_stress`, `fragile_tissue`, and `narrow_bottleneck` are the main robustness suites currently worth reporting
- the earlier `high_bacteria_stochasticity` output should be treated as provisional because the bacteria stochasticity override was previously blocked by a module-level singleton and now needs a rerun

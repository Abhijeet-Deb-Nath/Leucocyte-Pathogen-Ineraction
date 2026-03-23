# Robustness Artifacts

This directory stores curated paper-track robustness evaluations for the frozen learned controller.

Current intended layout:

- `results/robustness/hier_dagger_main/`

Contents of the curated bundle:

- aggregate robustness summary
- suite manifest
- per-suite summary CSVs
- per-suite detailed CSVs

Current interpretation:

- the frozen hierarchical `BC + DAgger` controller remains competitive or stronger than the heuristic in all currently reported stress settings
- `high_bacteria_stochasticity`, `hotspot_overload`, and `recruitment_delay_stress` are the strongest robustness wins
- `fragile_tissue` and `narrow_bottleneck` are smaller but still supportive robustness results

The corrected robustness rerun replaced the earlier provisional stochasticity result, so `results/robustness/hier_dagger_main/` should be treated as the authoritative robustness bundle.

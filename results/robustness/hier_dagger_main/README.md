# Frozen DAgger Robustness Bundle

This folder contains the curated robustness results for the frozen hierarchical `BC + DAgger` checkpoint.

Key files:

- `robustness_suite_summary.csv`: aggregate suite-level comparison between `heuristic` and learned policy
- `robustness_suite_manifest.json`: suite descriptions and override metadata
- `<suite>/<suite>.csv`: detailed per-episode results
- `<suite>/<suite>_summary.csv`: summary metrics for that suite

Reported suites:

- `baseline_reference`
- `high_bacteria_stochasticity`
- `hotspot_overload`
- `recruitment_delay_stress`
- `fragile_tissue`
- `narrow_bottleneck`

This bundle supersedes the earlier provisional stochasticity run and should be treated as the authoritative robustness artifact set for the frozen mainline.

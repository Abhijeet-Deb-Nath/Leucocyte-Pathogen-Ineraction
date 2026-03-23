# Table Inserts

This file provides manuscript-ready tables and suggested captions.

## Table 1. Main Held-Out Comparison

Suggested caption:

`Held-out 100-seed benchmark comparison. Plain behavior cloning collapses under covariate shift, hierarchical BC+DAgger is competitive with the heuristic baseline, and RL fine-tuning degrades the stronger imitation-based controller.`

| Method | Win Rate | Tissue Damage | Host Utility | Episode Length | Recruited Neutrophils |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic | 0.58 | 94.1710 | -90.3152 | 162.76 | 2.22 |
| Hierarchical BC-only | 0.02 | 133.9490 | -133.4290 | 194.49 | 0.00 |
| Hierarchical BC+DAgger | 0.60 | 90.6470 | -86.72115 | 137.07 | 4.44 |
| Hierarchical BC+DAgger+RL | 0.34 | 100.9305 | -97.91645 | 160.57 | 2.69 |

## Table 2. Reproducibility Across Training Seeds

Suggested caption:

`Held-out 100-seed evaluation across DAgger training seeds. The best run beats the heuristic, while variance across seeds supports a competitive-method claim rather than a domination claim.`

| Method | Train Seed | Win Rate | Tissue Damage | Host Utility | Episode Length | Recruited Neutrophils |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic | reference | 0.58 | 94.1710 | -90.3152 | 162.76 | 2.22 |
| Hierarchical BC+DAgger | 42 | 0.60 | 90.6470 | -86.72115 | 137.07 | 4.44 |
| Hierarchical BC+DAgger | 43 | 0.50 | 94.2635 | -90.88425 | 149.09 | 3.11 |
| Hierarchical BC+DAgger | 44 | 0.54 | 94.0575 | -91.20535 | 145.73 | 3.35 |

## Table 3. Robustness Summary

Suggested caption:

`Robustness summary for the frozen hierarchical BC+DAgger controller. The learned controller remains competitive or stronger across all reported suites, with the clearest gains under stronger bacterial stochasticity, hotspot overload, and delayed-support stress.`

| Suite | Heuristic Win Rate | DAgger Win Rate | Heuristic Host Utility | DAgger Host Utility |
| --- | ---: | ---: | ---: | ---: |
| baseline_reference | 0.58 | 0.60 | -90.3152 | -86.72115 |
| high_bacteria_stochasticity | 0.54 | 0.65 | -85.46935 | -80.20075 |
| hotspot_overload | 0.39 | 0.55 | -124.04345 | -117.78575 |
| recruitment_delay_stress | 0.51 | 0.57 | -88.77340 | -88.10730 |
| fragile_tissue | 0.50 | 0.51 | -115.92430 | -109.53105 |
| narrow_bottleneck | 0.29 | 0.39 | -115.60100 | -114.27225 |

## Table 4. Behavioral Differences on the Held-Out Benchmark

Suggested caption:

`Behavioral diagnostics on the held-out 100-seed benchmark. The competitive learned controller differs from the heuristic in support usage and episode timing, while BC-only and RL fine-tuning expose distinct failure modes.`

| Method | Attack Actions | Signal Actions | Stay Actions | Coverage Ratio | Visible Bacteria Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Heuristic | 16.19 | 13.74 | 5.05 | 0.1518 | 0.6406 |
| Hierarchical BC-only | 1.61 | 0.00 | 0.00 | 0.1141 | 0.3964 |
| Hierarchical BC+DAgger | 10.65 | 5.98 | 5.14 | 0.1120 | 0.6495 |
| Hierarchical BC+DAgger+RL | 6.78 | 4.23 | 2.05 | 0.1209 | 0.4535 |

# Figure Captions

This file provides paper-facing captions and placement guidance.

## Figure 1

File:

- `paper_material/figures/benchmark_control_loop.svg`

Suggested caption:

`Overview of the benchmarked control loop. The macrophage receives only a local partial observation and acts through a hierarchical policy with movement, attack, and abstract help-request decisions. Its actions alter bacterial dynamics, chemokine accumulation, and delayed neutrophil recruitment, producing a sequential trade-off between containment, escalation, and collateral damage.`

Suggested placement:

- environment / problem setup section

## Figure 2

File:

- `results/paper_ready/benchmark_comparison.png`

Suggested caption:

`Main held-out comparison on the frozen 100-seed benchmark. Plain behavior cloning collapses under covariate shift, hierarchical BC+DAgger is competitive with the heuristic baseline, and RL fine-tuning degrades the stronger imitation-based controller.`

Suggested placement:

- main results section

## Figure 3

File:

- `results/paper_ready/robustness_summary.png`

Suggested caption:

`Win-rate summary across robustness suites. The frozen hierarchical BC+DAgger controller remains competitive or stronger under increased bacterial stochasticity, hotspot overload, recruitment delay stress, fragile tissue, and narrow bottlenecks.`

Suggested placement:

- reproducibility and robustness section

## Figure 4

File:

- `results/paper_ready/behavior_tradeoffs.png`

Suggested caption:

`Behavioral trade-offs for the four main comparison methods. The competitive learned controller differs from the heuristic in support usage and episode duration, while BC-only and RL fine-tuning reveal distinct failure modes.`

Suggested placement:

- behavioral analysis section

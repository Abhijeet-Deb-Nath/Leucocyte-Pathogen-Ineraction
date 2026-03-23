# Paper Draft Outline

## Working Framing

This should be written as a benchmark/control paper, not as a new-algorithm paper and not as a biological-discovery paper.

Core claim:

- under partial observability, delayed support, and adaptive pathogen pressure, a hierarchical recurrent `BC + DAgger` controller becomes competitive with a strong interpretable heuristic and remains robust under stress

## Candidate Titles

1. `Partial-Observation Immune Control with Delayed Support: A Spatial Benchmark and Hierarchical Recurrent Baseline`
2. `Robust Macrophage Control under Delayed Immune Support in a Partially Observable Spatial Infection Simulator`
3. `Hierarchical Recurrent Imitation for Partially Observable Immune Control with Delayed Reinforcement`

## Abstract Skeleton

Paragraph 1:

- motivate immune control as a sequential decision problem under partial observability and delayed support
- position the environment as a spatial control benchmark rather than a clinical model

Paragraph 2:

- describe the benchmark ingredients:
  - locally sensing macrophage
  - adaptive bacteria
  - delayed neutrophil recruitment
  - tissue-damage / utility tradeoff

Paragraph 3:

- compare heuristic baseline to `BC-only`, hierarchical recurrent `BC + DAgger`, and `BC + DAgger + RL`
- report that `BC-only` fails, `BC + DAgger` is competitive, and RL fine-tuning degrades the stronger DAgger prior
- report robustness under stress suites for the frozen DAgger controller

Paragraph 4:

- state honest limitation:
  - same-information teacher-student setup
- position privileged-teacher extensions as future work

## Section Outline

### 1. Introduction

- local immune control is partially observed
- support is delayed and costly
- handcrafted rules are strong but limited
- contribution is a benchmark plus a first competitive learned controller

### 2. Environment / Benchmark

- state, observations, and action space
- macrophage objective
- bacteria dynamics
- delayed neutrophil support
- episode termination and host utility
- robustness suite definitions

### 3. Baselines And Learned Controller

- heuristic teacher / baseline
- hierarchical recurrent policy
- macro modes and `request_help` abstraction
- BC, DAgger, and RL-finetune variants

### 4. Experimental Protocol

- held-out 50-seed and 100-seed evaluations
- reproducibility across training seeds
- robustness suite settings
- metrics and behavioral diagnostics

### 5. Main Results

- main comparison table:
  - heuristic
  - BC-only
  - BC+DAgger
  - BC+DAgger+RL
- emphasize DAgger as the learned mainline

### 6. Reproducibility And Robustness

- training seed variance
- robustness suite summary
- highlight strongest robustness wins:
  - `high_bacteria_stochasticity`
  - `hotspot_overload`
  - `recruitment_delay_stress`

### 7. Behavioral Differences

- compare support usage, episode length, tissue damage, utility
- argue that the learned controller is not just matching final scores, but trading support and control differently

### 8. Limitations And Future Work

- same-information teacher-student setup
- current support mechanism is delayed-support, not deep coordination
- current benchmark is still custom and not yet a full benchmark family
- future direction:
  - privileged teacher
  - belief distillation
  - inferability-aware supervision

## Required Tables And Figures

### Required tables

1. Main comparison table
2. Reproducibility table
3. Robustness table

### Required figures

1. `results/paper_ready/benchmark_comparison.png`
2. `results/paper_ready/robustness_summary.png`
3. `results/paper_ready/behavior_tradeoffs.png`

## Current Blocking Items

The main comparison inputs are now complete.

Remaining work is on the documentation / packaging side rather than the research-discovery side:

- convert the outline into the full manuscript text
- turn the behavior-difference plots into a concise results subsection
- select 2-3 qualitative rollout examples for the appendix or demo material

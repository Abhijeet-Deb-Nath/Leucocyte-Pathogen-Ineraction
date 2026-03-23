# Claims And Evidence

This file is the claim-discipline guardrail for the paper.

Every central claim should map to a committed artifact.

## Claim 1

Claim:

- the benchmark is hard enough that plain behavior cloning is not a viable solution

Evidence:

- `results/final_eval/hier_bc_100ep_summary.csv`
- `results/paper_ready/main_comparison.csv`

Supporting numbers:

- `BC-only` win rate `0.02`
- `BC-only` tissue damage `133.9490`
- `BC-only` host utility `-133.4290`

Allowed wording:

- plain sequence-aware imitation collapses under distribution shift
- behavior cloning alone is inadequate in this partially observed benchmark

Do not write:

- BC is universally bad
- recurrence is useless

## Claim 2

Claim:

- hierarchical `BC + DAgger` is the first learned controller in this project that is competitive with the heuristic

Evidence:

- `results/final_eval/hier_dagger_100ep_summary.csv`
- `results/paper_ready/main_comparison.csv`

Supporting numbers:

- heuristic win rate `0.58`
- `BC + DAgger` win rate `0.60`
- heuristic tissue damage `94.1710`
- `BC + DAgger` tissue damage `90.6470`
- heuristic host utility `-90.3152`
- `BC + DAgger` host utility `-86.72115`

Allowed wording:

- competitive with the heuristic
- slightly better on the frozen held-out benchmark
- more tissue-preserving in the frozen best run

Do not write:

- universally superior
- clearly dominates the heuristic in all settings

## Claim 3

Claim:

- RL fine-tuning is not an improvement path in the current setup

Evidence:

- `results/final_eval/hier_rl_100ep_summary.csv`
- `results/paper_ready/main_comparison.csv`

Supporting numbers:

- `BC + DAgger + RL` win rate `0.34`
- `BC + DAgger + RL` tissue damage `100.9305`
- `BC + DAgger + RL` host utility `-97.91645`

Allowed wording:

- the current RL fine-tuning regime degrades the stronger DAgger controller
- RL is a negative ablation in the current paper

Do not write:

- RL cannot help this class of problems
- policy gradients are fundamentally unsuitable here

## Claim 4

Claim:

- the frozen DAgger controller remains competitive or stronger under the reported robustness suites

Evidence:

- `results/robustness/hier_dagger_main/robustness_suite_summary.csv`
- `results/paper_ready/robustness_table.csv`

Supporting suites:

- `high_bacteria_stochasticity`
- `hotspot_overload`
- `recruitment_delay_stress`
- `fragile_tissue`
- `narrow_bottleneck`

Allowed wording:

- competitive or stronger across all reported suites
- strongest gains under stochasticity, overload, and delayed-support stress

Do not write:

- universally robust to all possible distribution shifts
- guaranteed to generalize outside the current benchmark family

## Claim 5

Claim:

- the learned method is reproducibly competitive, but seed variance is real

Evidence:

- `results/final_eval/reproducibility_table.csv`
- `results/final_eval/REPRODUCIBILITY_SUMMARY.md`
- `results/paper_ready/reproducibility_table.csv`

Supporting numbers:

- train seed `42`: win rate `0.60`
- train seed `43`: win rate `0.50`
- train seed `44`: win rate `0.54`

Allowed wording:

- the best run beats the heuristic
- the average learned behavior is competitive and often more tissue-preserving
- seed variance prevents a domination claim

Do not write:

- the learned controller reliably outperforms the heuristic on every training seed

## Claim 6

Claim:

- the current project is a benchmark/control paper, not a new-algorithm paper

Evidence:

- method components are established techniques
- the benchmark/task formulation, negative ablations, and robustness suite are the main contribution class

Allowed wording:

- a benchmark/control study
- a partially observable spatial immune-control task with a first competitive learned baseline

Do not write:

- a fundamentally new imitation-learning method
- a novel RL algorithm

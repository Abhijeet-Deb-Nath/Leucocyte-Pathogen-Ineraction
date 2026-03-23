# Pathway 1 Immune-Pathogen Simulator

This repository is the cleaned paper-track codebase for a partially observable adversarial immune-control project.

Core question:

How should a locally sensing macrophage control a stochastic bacterial infection when escalation is delayed, costly, and only partially observed?

For the environment rules, see `detailed_game_rule.md`.
For the current project status, see `PROJECT_STATUS.md`.
For the frozen mainline and experiment interpretation, see `FROZEN_MAINLINE.md`, `RESULTS_SUMMARY.md`, `ROBUSTNESS_RESULTS.md`, and `EXPERIMENT_MATRIX.md`.

## Current Mainline

The repo now has one active learned-control path:

- `heuristic` as teacher and baseline
- hierarchical recurrent `BC + DAgger` as the main learned controller
- optional `BC + DAgger + RL` only as an ablation / unstable extension

The learned-controller stack currently includes:

- recurrent inference support in `agents/rl_agent.py`
- hierarchical policy and agent support in `agents/hierarchical_policy.py` and `agents/hierarchical_agent.py`
- partial-observation wrapper and action abstraction in `simulator/rl_interface.py`
- hierarchical behavior cloning, DAgger, and optional RL fine-tuning in `experiments/train_hierarchical.py`
- robustness evaluation in `experiments/run_robustness_suite.py`

The active action abstraction is:

- local movement
- attack
- abstract `request_help`

`request_help` is resolved inside the environment wrapper into concrete signal intensity based on local pressure, so the learned policy does not spend capacity on low/medium/high signal variants directly.

## Current Result Snapshot

Frozen checkpoint:

- `models/frozen/hier_dagger_main.pt`

Held-out 100-seed benchmark:

- heuristic:
  - `win_rate = 0.58`
  - `tissue_damage = 94.1710`
  - `host_utility = -90.3152`
- hierarchical `BC-only`:
  - `win_rate = 0.02`
  - `tissue_damage = 133.9490`
  - `host_utility = -133.4290`
- hierarchical `BC + DAgger`:
  - `win_rate = 0.60`
  - `tissue_damage = 90.6470`
  - `host_utility = -86.72115`
- hierarchical `BC + DAgger + RL`:
  - `win_rate = 0.34`
  - `tissue_damage = 100.9305`
  - `host_utility = -97.91645`

Final ablation status:

- `BC-only` fails badly under partial-observation compounding error
- `BC + DAgger` is the learned mainline
- `BC + DAgger + RL` degrades the stronger frozen DAgger policy and remains a negative ablation

Robustness summary:

- the frozen learned controller is competitive or stronger in all currently reported stress suites
- especially strong gains appear in `high_bacteria_stochasticity`, `hotspot_overload`, and `recruitment_delay_stress`

## What Was Removed

The repo no longer treats these as active paper paths:

- runtime MCTS / online planner control
- hidden planner fallback inside `Environment.step()`
- planner-based evaluation scripts and planner-specific config
- exploratory bacteria-mode and parameter-sweep scripts that depended on removed planner-era assumptions
- uncurated local CSV dumps that were blurring the project goal

These were removed because they were no longer promising relative to the hardware budget, GUI demo requirement, and current research direction.

## Repository Layout

- `agents/`: bacteria policy, heuristic teacher, learned-controller inference
- `simulator/`: environment, entities, RL wrapper
- `experiments/`: training, evaluation, robustness, reproducibility checks, simulation runner
- `visualization/`: GUI and plotting
- `models/`: local checkpoint space; `models/frozen/` is the stable home for the main learned checkpoint
- `results/final_eval/`: curated held-out evaluation and reproducibility summaries
- `results/robustness/`: curated robustness bundles for the frozen mainline
- `results/paper_ready/`: regenerated comparison tables and publication-facing figures

## Setup

```bash
python -m pip install -r requirements.txt
```

Optional editable install:

```bash
python -m pip install -e .
```

## Local Usage

Run the heuristic baseline in the GUI:

```bash
python -m experiments.run_simulation
```

Run the frozen learned checkpoint in the GUI:

```bash
python -m experiments.run_simulation --policy rl --rl-model-path models/frozen/hier_dagger_main.pt
```

Run a headless heuristic-vs-learned evaluation:

```bash
python -m experiments.evaluate_policies --episodes 20 --seed 123 --rl-model models/frozen/hier_dagger_main.pt --output-csv path/to/results.csv
```

Run the paper-track robustness suite:

```bash
python -m experiments.run_robustness_suite --rl-model models/frozen/hier_dagger_main.pt --output-dir results/robustness/hier_dagger_main
```

Run hierarchical BC-only bootstrap:

```bash
python -m experiments.train_hierarchical --stages bc --save-dir models/hier_bc --bc-pretrain-steps 4096 --bc-epochs 8 --bc-sequence-length 32
```

Run hierarchical BC + DAgger:

```bash
python -m experiments.train_hierarchical --stages bc dagger --save-dir models/hier_dagger_main
```

Run hierarchical BC + DAgger + RL fine-tuning:

```bash
python -m experiments.train_hierarchical --stages bc dagger rl_finetune --save-dir models/hier_rl_extension
```

## Artifact Policy

This repo intentionally does not keep arbitrary generated artifacts in version control.

Do not keep in git:

- exploratory checkpoints
- smoke-run folders
- temporary Colab outputs
- simulation history exports
- redundant downloaded bundles outside the curated results folders

Curated exceptions that are intentionally kept:

- `results/final_eval/**`
- `results/robustness/**`
- `results/paper_ready/**`

Heavy intermediate artifacts should still live in Google Drive or another external location.

## Practical Recommendation

Use the repo this way:

- local machine: correctness checks, GUI inspection, and frozen-checkpoint demonstration
- remote runtime: BC, DAgger, robustness evaluation, and any future controlled training extensions
- git repo: source code, documentation, and only the curated final result bundles

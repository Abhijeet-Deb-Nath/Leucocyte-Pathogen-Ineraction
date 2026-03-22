# Pathway 1 Immune-Pathogen Simulator

This repository is the cleaned paper-track codebase for a **partial-observation adversarial immune-control** project.

The central question is:

How should a locally sensing macrophage control a stochastic bacterial infection when escalation is delayed, costly, and only partially observed?

For the formal environment and interaction rules, see `detailed_game_rule.md`.
For the current research stage and next algorithmic milestones, see `PROJECT_STATUS.md`.
For the frozen mainline result and experiment structure, see `FROZEN_MAINLINE.md`, `RESULTS_SUMMARY.md`, and `EXPERIMENT_MATRIX.md`.

## Active Research Line

This repo now keeps one active host-control direction:
- `heuristic` only as an interpretable teacher and baseline
- one **learned macrophage controller** as the mainline
- partial observation as the only public benchmark path

The learned-controller stack currently includes:
- recurrent inference support in `agents/rl_agent.py`
- hierarchical policy / agent support in `agents/hierarchical_policy.py` and `agents/hierarchical_agent.py`
- partial-observation wrapper and action abstraction in `simulator/rl_interface.py`
- hierarchical behavior-cloning, DAgger, and optional RL fine-tuning in `experiments/train_hierarchical.py`

The active action abstraction is:
- local movement
- attack
- abstract `request_help`

`request_help` is resolved inside the environment wrapper into concrete signal intensity based on local pressure, so the learned policy does not spend capacity gaming low/medium/high signal variants directly.

## What Was Removed

The repo no longer carries these as active paper paths:
- runtime MCTS / online planner control
- hidden planner fallback inside `Environment.step()`
- planner-based evaluation scripts and planner-specific config
- exploratory bacteria-mode and parameter-sweep scripts that depended on the removed default planner
- versioned local evaluation CSV snapshots that were blurring the project goal

These were removed because they were no longer promising relative to the hardware budget, GUI demo requirement, and current research direction.

## Current Mainline Plan

The repo is now aligned around this sequence:

1. Train a hierarchical recurrent partial-observation controller from heuristic demonstrations.
2. Improve it with DAgger on learner-visited states.
3. Use RL fine-tuning only as a later extension if it improves the frozen DAgger model without destabilizing it.

The current frozen mainline is:
- hierarchical recurrent `BC + DAgger`
- checkpoint path: `models/frozen/hier_dagger_main.pt`

The current high-level modes are:
- `engage`
- `request_help`
- `follow_chem`
- `patrol`

Current method status:
- `heuristic`: teacher and baseline
- `hierarchical BC + DAgger`: main paper-track learned controller
- `hierarchical BC + DAgger + RL`: unstable extension / ablation, not the headline result

Headline held-out result:
- on the current `100`-seed benchmark, the frozen DAgger controller is slightly better than the heuristic on `win_rate`, `tissue_damage`, and `host_utility`
- full details are recorded in `RESULTS_SUMMARY.md`

## Repository Layout

- `agents/`: bacteria policy, heuristic teacher, learned-controller inference
- `simulator/`: environment, entities, RL wrapper
- `experiments/`: training, evaluation, reproducibility checks, simulation runner
- `visualization/`: GUI and plotting
- `models/`: local checkpoint space; `models/frozen/` is the stable home for the main learned checkpoint
- `results/`: final evaluation summaries and minimal paper-track artifacts

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

Run a learned checkpoint in the GUI:

```bash
python -m experiments.run_simulation --policy rl --rl-model-path path/to/checkpoint
```

Run a headless heuristic-vs-learned evaluation:

```bash
python -m experiments.evaluate_policies --episodes 20 --seed 123 --rl-model path/to/checkpoint --output-csv path/to/results.csv
```

Run the paper-track robustness suite:

```bash
python -m experiments.run_robustness_suite --rl-model models/frozen/hier_dagger_main.pt --output-dir results/robustness
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

This repo intentionally does **not** keep generated training and evaluation artifacts in version control.

Do not keep in git:
- checkpoints
- smoke-run CSVs
- evaluation CSVs
- simulation history exports
- remote experiment outputs

Store heavy artifacts in Google Drive or another external location and only bring back specific frozen checkpoints and final summaries when you need local inspection.

## Practical Recommendation

Use the repo this way:
- local machine: correctness checks, GUI inspection, and frozen-checkpoint demonstration
- remote runtime: BC, DAgger, robustness evaluation, and any future controlled training extensions
- git repo: source code, documentation, and the frozen mainline summaries

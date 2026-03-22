# Pathway 1 Immune-Pathogen Simulator

This repository is the cleaned paper-track codebase for a **partial-observation adversarial immune-control** project.

The central question is:

How should a locally sensing macrophage control a stochastic bacterial infection when escalation is delayed, costly, and only partially observed?

For the formal environment and interaction rules, see `detailed_game_rule.md`.
For the current research stage and next algorithmic milestones, see `PROJECT_STATUS.md`.

## Active Research Line

This repo now keeps one active host-control direction:
- `heuristic` only as an interpretable teacher and baseline
- one **learned macrophage controller** as the mainline
- partial observation as the only public benchmark path

The learned-controller stack currently includes:
- recurrent inference support in `agents/rl_agent.py`
- partial-observation wrapper and action abstraction in `simulator/rl_interface.py`
- sequence-aware behavior-cloning bootstrap and optional PPO fine-tuning in `experiments/train_ppo.py`

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

1. Train a recurrent partial-observation controller from expert demonstrations.
2. Improve it with better supervision on the learner's own visited states.
3. Add RL fine-tuning only after the imitation path becomes behaviorally competent.

The next planned algorithmic upgrades are:
- higher-level mode supervision such as `engage`, `request_help`, `follow_chem`, `patrol`
- DAgger-style dataset aggregation on learner-visited states
- optional RL fine-tuning after the imitation policy stops collapsing

## Repository Layout

- `agents/`: bacteria policy, heuristic teacher, learned-controller inference
- `simulator/`: environment, entities, RL wrapper
- `experiments/`: training, evaluation, reproducibility checks, simulation runner
- `visualization/`: GUI and plotting
- `models/`: local checkpoint scratch space only; not versioned with artifacts
- `results/`: intentionally documentation-only inside git; generated outputs should live outside version control

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

Run BC-only bootstrap:

```bash
python -m experiments.train_ppo --obs-mode partial_state --recurrent --timesteps 0 --bc-pretrain-steps 4096 --bc-epochs 8 --bc-sequence-length 32 --save-dir models/recurrent_bc
```

Run BC bootstrap plus PPO fine-tuning:

```bash
python -m experiments.train_ppo --obs-mode partial_state --recurrent --timesteps 30000 --save-dir models/recurrent_mainline
```

## Artifact Policy

This repo intentionally does **not** keep generated training and evaluation artifacts in version control.

Do not keep in git:
- checkpoints
- smoke-run CSVs
- evaluation CSVs
- simulation history exports
- remote experiment outputs

Store those artifacts in Google Drive or another external location and only bring back specific files when you need local inspection.

## Practical Recommendation

Use the repo this way:
- local machine: correctness checks, GUI inspection, short smoke checks
- remote runtime: longer behavior-cloning and RL training
- git repo: source code, documentation, and the current algorithmic path only

# Pathway 1 Immune-Pathogen Simulator

This repository is the paper-track codebase for a **partial-observation adversarial immune control** project.

The core question is:

How should a locally sensing macrophage control an acute bacterial infection when escalation is delayed and costly?

The project studies that question in a compartmentalized Pathway 1 simulator with:
- adaptive bacteria
- delayed neutrophil recruitment
- host utility and tissue-damage trade-offs
- multiple macrophage control methods under the same environment

For the formal game and mechanistic rules, see [detailed_game_rule.md](/C:/Users/Ankon/Desktop/Projects/AI/detailed_game_rule.md).
For the current research stage and next experimental gate, see [PROJECT_STATUS.md](/C:/Users/Ankon/Desktop/Projects/AI/PROJECT_STATUS.md).

## Research Focus

This repo is now intentionally focused on:
- **partial observation** as the main benchmark
- **heuristic vs lightweight MCTS vs RL**
- **recurrent PPO** as the main learning direction

This repo is no longer organized around:
- full-visibility results as the main story
- broad experimental sprawl
- keeping large local checkpoints and temporary smoke outputs in versioned state

## Methods

Current macrophage controllers:
- `heuristic`: rule-based local controller in [heuristic.py](/C:/Users/Ankon/Desktop/Projects/AI/agents/heuristic.py)
- `mcts`: lightweight local rollout planner in [mcts.py](/C:/Users/Ankon/Desktop/Projects/AI/agents/mcts.py)
- `rl`: PPO / recurrent PPO inference in [rl_agent.py](/C:/Users/Ankon/Desktop/Projects/AI/agents/rl_agent.py)

Current RL stack:
- feedforward PPO baseline
- masked recurrent PPO integrated as the new mainline
- shared observation/extractor path in [rl_interface.py](/C:/Users/Ankon/Desktop/Projects/AI/simulator/rl_interface.py)

## Repository Layout

- `agents/`: host and bacteria policies
- `simulator/`: environment, entities, RL wrapper
- `experiments/`: training, evaluation, sweeps, reproducibility checks
- `visualization/`: GUI and plotting
- `models/`: local output directory for checkpoints; large model artifacts are not kept in the cleaned repo
- `results/`: generated CSVs and experiment outputs

Results subfolders:
- `results/smoke/`: temporary smoke runs
- `results/medium_eval/`: intermediate evaluations
- `results/final_eval/`: paper-facing kept evaluation snapshots
- `results/sweeps/`: parameter sweeps
- `results/simulations/`: run history exports
- `results/archive/`: old artifacts if you choose to keep them locally

## Artifact Policy

This cleaned repo is intentionally light on generated artifacts.

Keep in the repo:
- source code
- dependency files
- documentation
- only a very small number of paper-facing summary CSVs

Do **not** keep in the repo long-term:
- temporary smoke checkpoints
- checkpoint folders
- eval logs
- transient local probe results
- large remote-training artifacts

Store large training outputs in Google Drive or another external artifact store.

## Current Kept Evaluation Snapshot

The current kept local baseline snapshot is:
- [partial_mainline_best_current_5ep.csv](/C:/Users/Ankon/Desktop/Projects/AI/results/final_eval/partial_mainline_best_current_5ep.csv)
- [partial_mainline_best_current_5ep_summary.csv](/C:/Users/Ankon/Desktop/Projects/AI/results/final_eval/partial_mainline_best_current_5ep_summary.csv)

That snapshot is the last kept feedforward partial-observation reference point:
- heuristic: `0.6` win rate
- mcts: `0.4` win rate
- rl: `0.4` win rate

These are not final paper numbers. They are the local reference baseline before the recurrent remote-training phase.

## Setup

Create and activate a Python environment, then install:

```bash
python -m pip install -r requirements.txt
```

Optional editable install:

```bash
python -m pip install -e .
```

## Local Usage

Run a single simulation with GUI:

```bash
python -m experiments.run_simulation
```

Run headless:

```bash
python -m experiments.run_simulation --headless
```

Run policy evaluation:

```bash
python -m experiments.evaluate_policies --episodes 20 --seed 123 --rl-model models/paper_track_partial_recurrent/ppo_macrophage_partial_state_recurrent_best --obs-mode partial_state --output-csv results/final_eval/policy_eval_partial_state_recurrent.csv
```

Run recurrent PPO locally only for short smoke checks:

```bash
python -m experiments.train_ppo --obs-mode partial_state --recurrent --timesteps 5000 --save-dir models/paper_track_partial_recurrent
```

## Remote Training Direction

The next serious training phase should run remotely, not on the local machine.

Recommended remote mainline:

```bash
python -m experiments.train_ppo --obs-mode partial_state --recurrent --timesteps 150000 --save-dir /content/runs/paper_track_partial_recurrent
```

Why remote now:
- masked recurrent PPO is structurally integrated
- invalid-action fallback collapse was fixed locally
- remaining work is learning quality, not wiring correctness

## Google Colab Workflow

Minimal workflow:

1. Open Google Colab and enable a GPU runtime if available.
2. Clone or upload the repo into `/content`.
3. Install dependencies with `pip install -r requirements.txt`.
4. Train in `/content`, not directly in Drive.
5. Copy only the important output artifacts to Drive after each run.

The first remote experiment should be a **moderate recurrent smoke run**, not a final large run.

## Development Notes

### Mainline Benchmark

The intended paper-track comparison is:
- heuristic
- lightweight MCTS
- feedforward PPO baseline
- recurrent PPO mainline

### Full-State Policy

Full-state experiments are not part of the paper-track mainline.
If legacy full-state code paths still exist, treat them as debugging leftovers or oracle-only support, not as the main research result.

### Action API

`Environment.get_macrophage_actions()` supports:
- `("move", (dx, dy))`
- `("attack", None)`
- `("signal_low", None)`
- `("signal_medium", None)`
- `("signal_high", None)`

### Reproducibility

Use:

```bash
python -m experiments.seed_repro_check
```

## Practical Recommendation

Use the repo this way:
- local machine: correctness checks, traces, GUI inspection
- remote runtime: recurrent PPO training and multi-seed evaluation
- repo: code, docs, and a small number of kept paper-facing summaries

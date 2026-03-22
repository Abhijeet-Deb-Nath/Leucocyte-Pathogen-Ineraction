# Pathway 1 Immune-Pathogen Simulator

This repository now implements the **Pathway 1** direction: an acute alveolar hotspot simulation with compartmentalized infection, delayed neutrophil recruitment, and host utility trade-offs.

This README is an engineering guide for running and extending the codebase.
For formal mechanics and detailed game design, see `detailed_game_rule.md`.

## What This Repository Is For

Use this project to:
- run Pathway 1 episodes and inspect outcomes (`winner`, `tissue_damage`, `host_utility`)
- compare behavior regimes through repeatable experiment scripts
- prototype new immune or bacterial policies without rewriting the full environment

Use this project less for:
- broad multi-pathway experimentation (the codebase is intentionally focused on Pathway 1)
- deriving the full biological rationale from README alone (that is in `detailed_game_rule.md`)

## Repository Map

- `config.py`: central knobs for world geometry, agent behavior, recruitment, damage, and planning budget.
- `simulator/entities.py`: entity dataclasses (`Macrophage`, `Bacteria`, `Neutrophil`).
- `simulator/environment.py`: core simulation loop, compartment map, fields, state transitions, recruitment, scoring.
- `agents/heuristic.py`: default host policy with contain-vs-recruit behavior.
- `agents/mcts.py`: rollout planner for host action selection.
- `agents/bacteria_adaptive.py`: local-context bacterial movement policy.
- `experiments/run_simulation.py`: single-run entrypoint with output artifacts.
- `experiments/compare_bacteria_modes.py`: compares adaptive bacteria stochasticity regimes.
- `experiments/param_sweep.py`: sweeps key Pathway 1 parameters.
- `visualization/pyqt_gui.py`: interactive PyQt viewer.
- `visualization/plots.py`: performance and heatmap plotting utilities.
- `visualization/animation.py`: lightweight matplotlib grid visualizer.

## Source and Artifact Layout

- Source code lives in `agents/`, `simulator/`, `visualization/`, and `experiments/`.
- Trained models live in `models/`.
- Generated CSVs and other run results go in `results/`.
- Use the `results/` subfolders to keep artifacts organized:
  - `results/smoke/`
  - `results/medium_eval/`
  - `results/final_eval/`
  - `results/sweeps/`
  - `results/simulations/`
  - `results/archive/`
- Module execution commands stay the same (`python -m ...` from repo root).

## Environment Setup

1. Create and activate a Python environment.
2. Install dependencies from repo root:

```bash
python -m pip install -r requirements.txt
```

Optional editable install (clean package-style local use):

```bash
python -m pip install -e .
```

Note:
- All experiment entrypoints below are shown as module commands (`python -m ...`) to avoid any `sys.path` hacks.

## Quick Start

Run one episode with GUI (default):

```bash
python -m experiments.run_simulation
```

Run headless only when needed for automation:

```bash
python -m experiments.run_simulation --headless
```

Organized output location:
- `results/simulations/simulation_history_pathway1.csv`

Run interactive GUI:

```python
from simulator.environment import Environment
from visualization.pyqt_gui import launch_gui

env = Environment()
launch_gui(env)
```

## Experiment Workflows

Compare bacterial policy noise regimes:

```bash
python -m experiments.compare_bacteria_modes
```

Artifacts should be kept under `results/`.

Sweep delayed recruitment and neutrophil collateral damage:

```bash
python -m experiments.param_sweep
```

Artifact location:
- `results/sweeps/pathway1_param_sweep_results.csv`

## PPO and Policy Commands

Install:

```bash
python -m pip install -r requirements.txt
```

Mainline training is partial-observation PPO:

```bash
python -m experiments.train_ppo --obs-mode partial_state --timesteps 150000 --save-dir models/paper_track_partial
```

Research-quality partial-observation training can also use recurrent PPO:

```bash
python -m experiments.train_ppo --obs-mode partial_state --recurrent --timesteps 150000 --save-dir models/paper_track_partial_recurrent
```

Notes:
- recurrent PPO uses `sb3-contrib`
- recurrent PPO now supports the same observation/extractor stack and explicit action masking

Evaluate policies (heuristic vs mcts vs rl):

```bash
python -m experiments.evaluate_policies --episodes 20 --seed 123 --rl-model models/paper_track_partial/ppo_macrophage_partial_state_best --obs-mode partial_state --output-csv results/final_eval/policy_eval_partial_state.csv
```

For recurrent checkpoints, point `--rl-model` at the recurrent artifact name:

```bash
python -m experiments.evaluate_policies --episodes 20 --seed 123 --rl-model models/paper_track_partial_recurrent/ppo_macrophage_partial_state_recurrent_best --obs-mode partial_state --output-csv results/final_eval/policy_eval_partial_state_recurrent.csv
```

Run local RL inference (single headless episode):

```bash
python -m experiments.run_simulation --headless --verbose --policy rl --rl-model-path models/paper_track_partial/ppo_macrophage_partial_state_best --rl-obs-mode partial_state
```

Run local RL inference with GUI:

```bash
python -m experiments.run_simulation --policy rl --rl-model-path models/paper_track_partial/ppo_macrophage_partial_state_best --rl-obs-mode partial_state
```

The same command works for recurrent checkpoints; `RLMacrophageAgent` now carries recurrent hidden state across steps and resets it between episodes.

The project treats partial observation as the primary benchmark. Full-state models, if kept for debugging, should be considered supporting analysis only.

Current kept paper-track artifacts in this repo:
- `models/partial_mainline_bc_attack_smoke/ppo_macrophage_partial_state_final.zip`
- `results/final_eval/partial_mainline_best_current_5ep.csv`
- `results/final_eval/partial_mainline_best_current_5ep_summary.csv`

## Google Colab Bootstrap (Minimal)

Clone and install:

```bash
!git clone <YOUR_REPO_URL>
%cd AI
!python -m pip install -r requirements.txt
```

Optional: mount Drive and set model output path:

```python
from google.colab import drive
drive.mount('/content/drive')
MODEL_DIR = '/content/drive/MyDrive/pathway1_models'
```

Train in Colab and save directly to Drive:

```bash
!python -m experiments.train_ppo --obs-mode partial_state --timesteps 150000 --save-dir "$MODEL_DIR/paper_track_partial"
```

Evaluate using saved Drive model:

```bash
!python -m experiments.evaluate_policies --episodes 20 --seed 123 --rl-model "$MODEL_DIR/paper_track_partial/ppo_macrophage_partial_state_best" --obs-mode partial_state --output-csv "$MODEL_DIR/policy_eval_partial_state.csv"
```

## How to Tune Without Breaking the Model

Recommended tuning order:
1. `config.py` geometry (`WORLD_ROWS`, `WORLD_COLS`, `COMPARTMENT_SIZE`, bottlenecks).
2. Replication pressure (`BASE_REPLICATION_PROB`, local carrying capacity, nutrient thresholds).
3. Recruitment pacing (`NEUTROPHIL_RECRUITMENT_THRESHOLD`, `NEUTROPHIL_ARRIVAL_DELAY`).
4. Damage economics (`TISSUE_DAMAGE_*`, `HOST_UTILITY_ALPHA/BETA`).
5. Planning budget (`ROLLOUT_BUDGET`, `MCTS_SIM_DEPTH`).

Practical rule:
- change one subsystem at a time and keep episode outputs for before/after comparison.

## Development Notes

### Action API
`Environment.get_macrophage_actions()` currently supports:
- `("move", (dx, dy))`
- `("attack", None)`
- `("signal_low", None)`
- `("signal_medium", None)`
- `("signal_high", None)`

Any host policy should consume this API rather than hardcoding actions.

### Core Runtime Contract
If you modify environment internals, keep these stable:
- `env.step(...)`
- `env.clone()` for planner rollouts
- `env.history` keys used by plots and experiments
- `env.compute_host_utility()`
- `env.get_grid()` for visualization backends

### Performance Tips
- For faster debugging, set `ROLLOUT_BUDGET = 0` to force heuristic host policy.
- For reproducible tests, pass `seed` when creating `Environment(seed=...)`.

## Troubleshooting

If GUI fails to launch:
- verify `PyQt5` is installed in the active Python environment
- run headless workflows first (`experiments/run_simulation.py`)

If runs feel too slow:
- reduce `ROLLOUT_BUDGET`
- reduce `MCTS_SIM_DEPTH`
- use smaller world geometry while iterating

If outcomes look degenerate (always host win or always bacteria win):
- rebalance recruitment delay and collateral damage before changing many parameters at once

## Documentation Split

- `README.md` (this file): how to use and evolve the codebase.
- `detailed_game_rule.md`: authoritative mechanics and design specification.

This split is intentional to keep implementation guidance concise and avoid duplicating the rulebook.

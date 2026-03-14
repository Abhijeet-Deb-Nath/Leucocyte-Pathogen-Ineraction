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

## Environment Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install numpy pandas matplotlib PyQt5
```

Optional:
- `seaborn` is not required in the current code path.

## Quick Start

Run one episode with GUI (default):

```bash
python experiments/run_simulation.py
```

Run headless only when needed for automation:

```bash
python experiments/run_simulation.py --headless
```

Expected output in repo root:
- `simulation_history_pathway1.csv`

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
python experiments/compare_bacteria_modes.py
```

Artifacts:
- `pathway1_bacteria_policy_comparison.csv`
- `pathway1_bacteria_policy_comparison.png`

Sweep delayed recruitment and neutrophil collateral damage:

```bash
python experiments/param_sweep.py
```

Artifact:
- `pathway1_param_sweep_results.csv`

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
- `("signal", None)`

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

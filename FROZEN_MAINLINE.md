# Frozen Mainline

## Current Main Learned Controller

The current frozen learned mainline is:

- `hierarchical recurrent BC + DAgger`

Checkpoint:

- `models/frozen/hier_dagger_main.pt`

## Supporting Evaluation Files

Curated held-out evaluation files live in `results/final_eval/`:

- `hier_dagger_training_summary.json`
- `hier_dagger_50ep.csv`
- `hier_dagger_50ep_summary.csv`
- `hier_dagger_100ep.csv`
- `hier_dagger_100ep_summary.csv`
- `hier_dagger_final_eval_summary.csv`

Curated robustness files live in:

- `results/robustness/hier_dagger_main/`

## Why This Is The Mainline

Held-out 100-seed benchmark:

- heuristic:
  - `win_rate = 0.58`
  - `tissue_damage = 94.1710`
  - `host_utility = -90.3152`
- hierarchical `BC + DAgger`:
  - `win_rate = 0.60`
  - `tissue_damage = 90.6470`
  - `host_utility = -86.72115`

The frozen controller also remains competitive or better in the current robustness suites:

- `high_bacteria_stochasticity`
- `hotspot_overload`
- `recruitment_delay_stress`
- `fragile_tissue`
- `narrow_bottleneck`

## Method Interpretation

- `heuristic`: teacher and reference baseline
- `hierarchical BC + DAgger`: main paper-track method
- `hierarchical BC + DAgger + RL`: ablation / unstable extension, not the mainline

## How To Use This Checkpoint

Use the frozen mainline for:

- GUI demonstration
- final comparison tables
- robustness reporting
- paper discussion and screenshots

Do not replace it casually with longer or more exploratory training runs unless a new run is clearly stronger on held-out evaluation and remains stable across reproducibility checks.

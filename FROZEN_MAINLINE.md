# Frozen Mainline

## Current Main Learned Controller

The current frozen learned mainline is:

- `hierarchical recurrent BC+DAgger`

Checkpoint name:

- `models/frozen/hier_dagger_main.pt`

## Supporting Evaluation Files

Keep the following in `results/final_eval/`:

- `hier_dagger_training_summary.json`
- `hier_dagger_50ep.csv`
- `hier_dagger_50ep_summary.csv`
- `hier_dagger_100ep.csv`
- `hier_dagger_100ep_summary.csv`
- `hier_dagger_final_eval_summary.csv`

## Interpretation

This is the first learned controller that is competitive with the heuristic under the frozen partially observable adversarial benchmark.

Current interpretation of methods:

- `heuristic`: teacher and reference baseline
- `hierarchical BC+DAgger`: main paper-track method
- `hierarchical BC+DAgger+RL`: unstable extension, not the mainline

## Next Research Phase

Use this frozen model for:

- GUI demonstration
- reproducibility experiments
- robustness experiments
- paper tables and discussion

Current robustness artifacts:

- `results/robustness/hier_dagger_main/`

Do not replace it casually with new training runs unless those runs are clearly stronger on held-out evaluation.

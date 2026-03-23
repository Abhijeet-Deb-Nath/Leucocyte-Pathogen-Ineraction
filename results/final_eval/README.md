# Final Evaluation Artifacts

This directory stores the curated paper-track evaluation outputs for the final comparison table, reproducibility summary, and frozen learned-controller reference.

Main comparison artifacts:

- `hier_bc_100ep.csv`
- `hier_bc_100ep_summary.csv`
- `hier_dagger_100ep.csv`
- `hier_dagger_100ep_summary.csv`
- `hier_rl_100ep.csv`
- `hier_rl_100ep_summary.csv`

Additional curated mainline artifacts:

- `hier_dagger_training_summary.json`
- `hier_dagger_50ep.csv`
- `hier_dagger_50ep_summary.csv`
- `hier_dagger_final_eval_summary.csv`
- `reproducibility_table.csv`

Interpretation of the current comparison table:

- `heuristic` = teacher and baseline
- `hierarchical BC-only` = negative ablation that fails under distribution shift
- `hierarchical BC+DAgger` = main learned method
- `hierarchical BC+DAgger+RL` = negative ablation that degrades the DAgger prior

The current research status is therefore:

- heuristic = teacher and baseline
- hierarchical recurrent BC+DAgger = main learned method
- RL fine-tuning = ablation / unstable extension

Keep this directory focused on final or near-final summaries, not transient smoke outputs from abandoned branches.

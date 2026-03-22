# Final Evaluation Artifacts

This directory stores the final paper-track evaluation outputs for the frozen learned controller and key baselines.

Recommended files for the current mainline:

- `hier_dagger_training_summary.json`
- `hier_dagger_50ep.csv`
- `hier_dagger_50ep_summary.csv`
- `hier_dagger_100ep.csv`
- `hier_dagger_100ep_summary.csv`
- `hier_dagger_final_eval_summary.csv`

These files correspond to the frozen hierarchical BC+DAgger controller.

The current research status is:

- heuristic = teacher and baseline
- hierarchical recurrent BC+DAgger = main learned method
- RL fine-tuning = ablation / unstable extension

Keep this directory focused on final or near-final summaries, not transient smoke outputs from abandoned branches.

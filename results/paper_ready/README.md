# Paper-Ready Artifacts

This directory stores generated tables and figures used for paper drafting.

The writing-side interpretation of these artifacts lives in `paper_material/`.

Build them with:

```bash
python -m experiments.build_paper_artifacts
```

Inputs:

- `results/final_eval/`
- `results/robustness/hier_dagger_main/`

Expected outputs:

- `main_comparison.csv`
- `reproducibility_table.csv`
- `robustness_table.csv`
- `benchmark_comparison.png`
- `robustness_summary.png`
- `behavior_tradeoffs.png`
- `missing_artifacts.txt`

`missing_artifacts.txt` is the authoritative indicator of whether the main 4-row comparison table is complete.

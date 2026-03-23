# Results Directory

This directory now contains a small set of curated paper-track result bundles.

Versioned subdirectories:

- `results/final_eval/`
- `results/robustness/`
- `results/paper_ready/`

These are intentional exceptions for:

- frozen held-out evaluation summaries
- curated robustness outputs
- generated paper tables and figures

The narrative interpretation of these artifacts should live in `paper_material/`, not inside the raw result bundles.

Do not keep in git:

- ad hoc smoke outputs
- temporary Colab folders
- duplicate downloaded bundles outside curated result folders
- simulation history exports
- arbitrary exploratory CSV dumps

Anything not part of the curated paper path should stay external or be deleted after inspection.

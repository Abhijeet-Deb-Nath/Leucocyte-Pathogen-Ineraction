# Paper Polishing Next Steps

## Current State

Already curated and ready:

- frozen hierarchical `BC + DAgger` checkpoint
- held-out DAgger evaluation
- held-out `BC-only` `100`-seed summary
- held-out `BC + DAgger + RL` `100`-seed summary
- reproducibility summary for training seeds `42`, `43`, and `44`
- corrected robustness suite

## Local Next Step

Build paper-ready artifacts from the current curated bundle:

```bash
python -m experiments.build_paper_artifacts
```

Then inspect:

- `results/paper_ready/missing_artifacts.txt`

At the current repo state, that file should report that all expected curated artifacts are present.

## Colab / Drive Need

No additional Colab or Drive work is required for the current polishing phase.

The only reason to return to remote compute now would be a deliberate new experiment outside the current paper scope.

## Submission-Focused Goal

The current paper should aim to deliver:

- main comparison table
- reproducibility table
- robustness table
- lightweight behavior-difference analysis
- benchmark/control framing with honest limitations

Immediate work is now:

1. tighten the docs around the completed four-row comparison
2. write the benchmark/control paper draft
3. reuse `results/paper_ready/` for tables and figures

This phase should not introduce:

- new agent families
- privileged teachers
- environment redesign
- new algorithmic pivots

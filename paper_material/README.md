# Paper Material

This folder is the canonical paper package for the current project.

The paper should be written as a benchmark/control paper:

- not as a new-algorithm paper
- not as a biological-discovery paper
- not as a translational clinical claim

The current paper claim is narrower and defensible:

- a partially observable spatial immune-control benchmark with delayed support and adaptive pathogen pressure
- a strong interpretable heuristic baseline
- a hierarchical recurrent `BC + DAgger` controller that becomes competitive under held-out evaluation
- a robustness suite showing that the frozen learned controller remains strong under stress

## Folder Contents

- `MANUSCRIPT_DRAFT.md`
  - working paper text with abstract, section framing, and contribution language
- `PAPER_DRAFT_OUTLINE.md`
  - compact structure of the manuscript
- `THEORY_AND_FORMULATION.md`
  - POMDP formalization, action abstraction, objective, and imitation-learning rationale
- `METRIC_DEFINITIONS.md`
  - exact benchmark and behavior metric definitions used in evaluation
- `POSITIONING_AND_RESEARCH_GAP.md`
  - what gap the paper fills, what it does not claim, and how to position it against nearby literature
- `CLAIMS_AND_EVIDENCE.md`
  - claim-by-claim mapping from manuscript language to result artifacts
- `FIGURE_CAPTIONS.md`
  - paper-facing captions and placement guidance for the main figures
- `TABLE_INSERTS.md`
  - manuscript-ready tables and caption guidance derived from the curated result bundle
- `SUBMISSION_STRATEGY.md`
  - realistic venue fit and presentation expectations
- `PAPER_POLISHING_NEXT_STEPS.md`
  - remaining packaging work for the current manuscript
- `figures/`
  - conceptual diagram assets that belong to the paper package rather than raw results

## Source Artifacts

Paper numbers should be pulled from:

- `results/paper_ready/main_comparison.csv`
- `results/paper_ready/reproducibility_table.csv`
- `results/paper_ready/robustness_table.csv`
- `results/paper_ready/benchmark_comparison.png`
- `results/paper_ready/robustness_summary.png`
- `results/paper_ready/behavior_tradeoffs.png`

## Working Order

1. Use `CLAIMS_AND_EVIDENCE.md` to constrain what the paper is allowed to claim.
2. Use `THEORY_AND_FORMULATION.md` when writing the environment, method, and experimental protocol sections.
3. Use `TABLE_INSERTS.md` and `FIGURE_CAPTIONS.md` when assembling the main results sections.
4. Use `MANUSCRIPT_DRAFT.md` as the paper-writing base.
5. Use `SUBMISSION_STRATEGY.md` only after the draft is complete and stable.

## Scope Discipline

This folder is for the current paper only.

Do not mix in:

- privileged-teacher experiments
- new teacher-student branches
- environment-family redesign
- method pivots that are not part of the current result bundle

Those belong to a later project branch.

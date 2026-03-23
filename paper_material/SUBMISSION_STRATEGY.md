# Submission Strategy

## 1. Paper Class

The current manuscript is best treated as:

- a benchmark/control paper
- an imitation-learning application paper with strong ablations
- an AI-for-science flavored sequential-decision paper

It is not a clean fit for a main-track deep-learning novelty paper.

## 2. Presentation Standards To Follow

The safest presentation standard is:

1. clearly define the learning task
2. clearly define the evaluation protocol
3. support every central claim with empirical evidence
4. state limitations explicitly

This matches formal review expectations in strong ML journals.

- JMLR reviewer guidance emphasizes:
  - clear goals and task definition
  - enough detail to replicate
  - adequate empirical or theoretical evaluation
  - significance supported by evidence
  - https://www.jmlr.org/reviewer-guide.html

- JMLR scope explicitly includes:
  - formalization of new learning tasks
  - methods for assessing performance on those tasks
  - https://jmlr.org/history.html

- TMLR editorial scope also includes work on new learning tasks
  - https://jmlr.org/tmlr/editorial-policies.html

- NeurIPS Datasets and Benchmarks allows benchmark submissions that are environments rather than datasets
  - https://neurips.cc/Conferences/2025/DatasetsBenchmarks-FAQ

## 3. Realistic Venue Fit

Most realistic fit:

- TMLR
- JAIR
- JMLR if the benchmark/task formalization is written strongly enough

Stretch fit:

- NeurIPS Datasets and Benchmarks

Poor fit for the current version:

- NeurIPS / ICML / ICLR main track
- biology journals expecting validated mechanistic or translational realism

## 4. Why The Current Paper Should Not Overreach

The current result is publishable because it is coherent, not because it is maximal.

The main reasons to avoid overreach are:

- algorithmic novelty is modest
- biological realism is intentionally abstract
- the strongest result is empirical benchmark evidence, not a new theorem or architecture

Presenting the paper as a clean benchmark/control study improves the probability that reviewers will evaluate it on the right axis.

## 5. Submission Checklist

Before submission, ensure the manuscript includes:

- exact benchmark definition
- exact observation and action definitions
- exact metric definitions
- full four-row ablation table
- reproducibility table
- robustness table
- clear limitations section
- code/release instructions consistent with the repo

## 6. Post-Submission Extension Path

The stronger privileged-teacher branch should remain a future paper, not a revision add-on to this manuscript.

Reason:

- it changes the central contribution class
- it would require asymmetric-observability analysis
- it would require a new method story rather than an incremental appendix

That branch should be opened only after the current paper draft is stable.

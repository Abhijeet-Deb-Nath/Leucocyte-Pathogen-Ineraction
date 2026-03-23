# Positioning And Research Gap

## 1. What Gap This Paper Actually Fills

The current gap is narrow but real.

Most nearby immune-control RL work emphasizes systemic intervention or treatment scheduling in mechanistic inflammatory models. That literature is useful, but it is not the same as local spatial control under partial observability.

This project instead focuses on:

- a locally sensing macrophage
- a spatial compartmental world with bottlenecks
- adaptive bacterial dynamics
- delayed neutrophil support
- explicit clearance-versus-damage accounting

That combination defines a different task class from global treatment optimization.

## 2. How To Position Against Nearby Literature

The correct positioning is:

- this is not the first use of RL or imitation in immune-inspired modeling
- it is a different control problem than systemic cytokine or treatment scheduling
- the contribution is the benchmark/task formulation and the ablation ladder, not a new generic learning algorithm

Relevant nearby literature:

- Ross and Bagnell, 2010, "Efficient Reductions for Imitation Learning"
  - https://proceedings.mlr.press/v9/ross10a.html
- Cockrell et al., 2022, "Preparing for the next pandemic: Simulation-based deep reinforcement learning to discover and test multimodal control of systemic inflammation using repurposed immunomodulatory agents"
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC9720328/

The difference from the systemic-inflammation line should be stated explicitly:

- those works study therapy-control or immunomodulation from different state and action abstractions
- the current work studies local macrophage decision-making under partial observability and delayed local reinforcement

## 3. Why The Current Paper Is Publishable Even Without New Algorithmic Novelty

The paper is publishable if it is presented as:

- formalization of a new learning task
- a reproducible benchmark/control study
- careful evaluation of baseline and learned controllers

This is not speculative. Major ML journals explicitly accept this type of contribution.

- JMLR includes "formalization of new learning tasks" and "methods for assessing performance on those tasks"
  - https://jmlr.org/history.html
- TMLR editorial scope includes work on new learning tasks
  - https://jmlr.org/tmlr/editorial-policies.html
- NeurIPS Datasets and Benchmarks explicitly allows environment-only benchmark submissions
  - https://neurips.cc/Conferences/2025/DatasetsBenchmarks-FAQ

## 4. What This Paper Should Not Pretend To Be

The current paper is not:

- a top-tier main-track deep-learning architecture paper
- a new imitation-learning theory paper
- a validated immunology simulator paper
- a privileged-teacher or belief-distillation paper

That boundary is important. Overstating the novelty would weaken reviewability.

## 5. What Makes The Current Story Scientifically Coherent

The manuscript has a clean internal logic.

1. The task is hard because the macrophage acts under partial observability, while escalation is delayed and costly.
2. The heuristic is strong enough that naive learned baselines must justify themselves.
3. `BC-only` fails, showing that simple supervised imitation is not enough.
4. `BC + DAgger` succeeds, showing that learner-state aggregation is the key mechanism.
5. RL fine-tuning fails, showing that unconstrained post-imitation optimization is not automatically beneficial.
6. Robustness suites show that the frozen mainline is not only a narrow seed-specific success.

That is a valid benchmark/control paper story.

## 6. Why The Teacher-Student Extension Is Not Part Of This Paper

The stronger privileged-teacher direction remains promising, but it is a different paper.

It would change the central question from:

- can an imitation-based recurrent controller become competitive in this benchmark?

to:

- how should privileged control be distilled into a partially observed student?

That is a genuine method pivot. Keeping it out of the current manuscript preserves clarity and protects the current paper from scope drift.

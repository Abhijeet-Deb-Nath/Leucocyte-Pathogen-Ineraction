# Experiment Matrix

## Purpose

This file defines the paper-track experiment structure for the current project.

The project should now be evaluated as a focused benchmark-and-method paper, not as a zoo of unrelated agent branches.

## Main Comparison Table

The primary table should include:

1. `heuristic`
2. `hierarchical BC-only`
3. `hierarchical BC + DAgger`
4. `hierarchical BC + DAgger + RL`

Primary metrics:

- `win_rate`
- `tissue_damage`
- `host_utility`
- `episode_length`
- `recruited_neutrophils`

Secondary behavioral metrics:

- `attack_actions`
- `signal_actions`
- `blind_signal_actions`
- `stay_actions`
- `coverage_ratio`
- `visible_bacteria_ratio`

## Current Interpretation

Expected narrative:

- `heuristic` is the interpretable expert baseline
- `BC-only` is a weak bootstrap
- `BC + DAgger` is the first competitive learned controller
- `BC + DAgger + RL` is currently unstable and should be reported as a failed or inconclusive extension

## Reproducibility Table

For the frozen benchmark, report at least:

- training seed `42`
- training seed `43`
- training seed `44`
- mean and standard deviation across these runs

Evaluate all on the same held-out `100`-seed benchmark.

Main reported reproducibility metrics:

- `win_rate`
- `tissue_damage`
- `host_utility`

## Robustness Table

After the standard benchmark, evaluate the frozen DAgger controller and heuristic under stress settings.

Recommended robustness suites:

1. higher bacteria stochasticity
2. harsher hotspot placement
3. recruitment-delay stress
4. tissue-damage sensitivity shift
5. bottleneck / doorway-heavy seeds

Use the same metrics as the main table.

Repository entrypoint for this phase:

```bash
python -m experiments.run_robustness_suite --rl-model models/frozen/hier_dagger_main.pt --output-dir results/robustness
```

## GUI Demonstration

The GUI demonstration should use:

- `heuristic`
- frozen `hierarchical BC + DAgger`

The GUI should not use:

- runtime MCTS
- RL-finetuned checkpoint
- transient smoke checkpoints

## Appendix / Negative Results

The appendix or discussion should acknowledge:

- flat PPO from scratch failed
- runtime MCTS was not a practical mainline under the benchmark and hardware constraints
- the first RL fine-tuning attempt destabilized the DAgger policy

These are useful scientific results, but they should not dominate the main paper framing.

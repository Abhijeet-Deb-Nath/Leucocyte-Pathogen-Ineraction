# Results Summary

## Current Main Result

The current main learned controller is:

- hierarchical recurrent `BC + DAgger`

Frozen checkpoint:

- `models/frozen/hier_dagger_main.pt`

This is the first learned controller in the project that is genuinely competitive with the heuristic under the frozen partially observable adversarial benchmark.

## Main Evaluation Snapshot

### 50-seed held-out evaluation

Heuristic:

- `win_rate = 0.64`
- `tissue_damage = 98.524`
- `host_utility = -93.6279`
- `episode_length = 158.34`
- `recruited_neutrophils = 2.44`

Hierarchical `BC + DAgger`:

- `win_rate = 0.62`
- `tissue_damage = 89.297`
- `host_utility = -84.9124`
- `episode_length = 144.46`
- `recruited_neutrophils = 4.60`

Interpretation:

- almost matched the heuristic on win rate
- improved tissue protection
- improved host utility
- ended episodes faster

### 100-seed held-out evaluation

Heuristic:

- `win_rate = 0.58`
- `tissue_damage = 94.1710`
- `host_utility = -90.3152`
- `episode_length = 162.76`
- `recruited_neutrophils = 2.22`

Hierarchical `BC + DAgger`:

- `win_rate = 0.60`
- `tissue_damage = 90.6470`
- `host_utility = -86.72115`
- `episode_length = 137.07`
- `recruited_neutrophils = 4.44`

Interpretation:

- slightly better win rate
- lower tissue damage
- better host utility
- faster resolution
- stronger use of reinforcement

This 100-seed result is the basis for freezing the current DAgger checkpoint as the project mainline.

## Reproducibility Snapshot

Three training seeds were examined for the hierarchical `BC + DAgger` method.

### Seed 42

- `win_rate = 0.60`
- `tissue_damage = 90.6470`
- `host_utility = -86.72115`

### Seed 43

- `win_rate = 0.50`
- `tissue_damage = 94.2635`
- `host_utility = -90.88425`

### Seed 44

- `win_rate = 0.54`
- `tissue_damage = 94.0575`
- `host_utility = -91.20535`

### Heuristic reference on the same 100-seed benchmark

- `win_rate = 0.58`
- `tissue_damage = 94.1710`
- `host_utility = -90.3152`

### Mean across learned runs

Approximate learned mean:

- `win_rate ≈ 0.55`
- `tissue_damage ≈ 92.99`
- `host_utility ≈ -89.60`

Interpretation:

- the learned method is reproducibly competitive
- seed-to-seed variance is real
- the best run beats the heuristic
- the average learned run is slightly worse on win rate but slightly better on tissue damage and host utility

This means the correct scientific claim is:

- the hierarchical `BC + DAgger` controller is **competitive and often more tissue-preserving**

The correct claim is **not**:

- the learned agent always dominates the heuristic

## RL Fine-Tuning Result

The current `BC + DAgger + RL` extension is not accepted as the mainline.

Reason:

- the initial RL fine-tuning pass degraded the DAgger policy
- held-out performance fell below the frozen DAgger checkpoint

So RL fine-tuning is currently an:

- ablation
- unstable extension
- future research branch

It is not the headline result.

## What More Training Means

More training is **not automatically better** from this point.

What likely helps:

- more careful DAgger data aggregation
- training multiple seeds and selecting by held-out validation
- conservative extensions that preserve the DAgger policy

What is not currently justified:

- longer blind RL fine-tuning
- returning to runtime MCTS
- changing the benchmark to make the learner look better

So the next research phase is:

- freeze the DAgger mainline
- run robustness experiments
- report reproducibility
- only then consider carefully constrained training extensions

## Robustness Follow-Up

Robustness evaluation has now been added as the next evidence layer.

Current status:

- multiple stress suites already support the frozen DAgger controller
- `high_bacteria_stochasticity` must be rerun after the bacteria-policy singleton bug fix

See:

- `ROBUSTNESS_RESULTS.md`
- `results/robustness/hier_dagger_main/`

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

## Final Ablation Snapshot

The full held-out `100`-seed comparison now has four rows.

### Heuristic

- `win_rate = 0.58`
- `tissue_damage = 94.1710`
- `host_utility = -90.3152`
- `episode_length = 162.76`
- `recruited_neutrophils = 2.22`

### Hierarchical `BC-only`

- `win_rate = 0.02`
- `tissue_damage = 133.9490`
- `host_utility = -133.4290`
- `episode_length = 194.49`
- `recruited_neutrophils = 0.00`

Interpretation:

- plain sequence-aware imitation is not enough under partial observability
- once the learner drifts off the teacher distribution, compounding error dominates
- this makes DAgger a necessary part of the current learned-control path

### Hierarchical `BC + DAgger`

- `win_rate = 0.60`
- `tissue_damage = 90.6470`
- `host_utility = -86.72115`
- `episode_length = 137.07`
- `recruited_neutrophils = 4.44`

Interpretation:

- this is the only learned controller that is clearly competitive with the heuristic
- it resolves episodes faster while preserving tissue better
- it remains the frozen mainline

### Hierarchical `BC + DAgger + RL`

- `win_rate = 0.34`
- `tissue_damage = 100.9305`
- `host_utility = -97.91645`
- `episode_length = 160.57`
- `recruited_neutrophils = 2.69`

Interpretation:

- RL fine-tuning did not improve the stronger DAgger controller
- the current RL extension is a negative ablation, not a viable replacement
- more blind RL training is not justified by the current evidence

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

- `win_rate ~ 0.55`
- `tissue_damage ~ 92.99`
- `host_utility ~ -89.60`

Interpretation:

- the learned method is reproducibly competitive
- seed-to-seed variance is real
- the best run beats the heuristic
- the average learned run is slightly worse on win rate but slightly better on tissue damage and host utility

The correct scientific claim is:

- the hierarchical `BC + DAgger` controller is competitive and often more tissue-preserving

The correct claim is not:

- the learned agent always dominates the heuristic

## RL Fine-Tuning Result

The current `BC + DAgger + RL` extension is not accepted as the mainline.

Reason:

- the RL fine-tuning ablation degraded the DAgger policy
- held-out performance fell below the frozen DAgger checkpoint

So RL fine-tuning is currently an:

- ablation
- unstable extension
- future research branch

It is not the headline result.

## Robustness Snapshot

The corrected robustness suite now supports the frozen learned controller across all reported stress settings.

### `baseline_reference`

- heuristic:
  - `win_rate = 0.58`
  - `tissue_damage = 94.1710`
  - `host_utility = -90.3152`
- learned:
  - `win_rate = 0.60`
  - `tissue_damage = 90.6470`
  - `host_utility = -86.72115`

### `high_bacteria_stochasticity`

- heuristic:
  - `win_rate = 0.54`
  - `tissue_damage = 88.8430`
  - `host_utility = -85.46935`
- learned:
  - `win_rate = 0.65`
  - `tissue_damage = 84.0700`
  - `host_utility = -80.20075`

### `hotspot_overload`

- heuristic:
  - `win_rate = 0.39`
  - `tissue_damage = 129.7790`
  - `host_utility = -124.04345`
- learned:
  - `win_rate = 0.55`
  - `tissue_damage = 123.2315`
  - `host_utility = -117.78575`

### `recruitment_delay_stress`

- heuristic:
  - `win_rate = 0.51`
  - `tissue_damage = 92.7105`
  - `host_utility = -88.7734`
- learned:
  - `win_rate = 0.57`
  - `tissue_damage = 92.3840`
  - `host_utility = -88.1073`

### `fragile_tissue`

- heuristic:
  - `win_rate = 0.50`
  - `tissue_damage = 119.5420`
  - `host_utility = -115.92430`
- learned:
  - `win_rate = 0.51`
  - `tissue_damage = 113.0116`
  - `host_utility = -109.53105`

### `narrow_bottleneck`

- heuristic:
  - `win_rate = 0.29`
  - `tissue_damage = 115.5295`
  - `host_utility = -115.60100`
- learned:
  - `win_rate = 0.39`
  - `tissue_damage = 115.6620`
  - `host_utility = -114.27225`

Interpretation:

- the learned controller is not only competitive on the frozen benchmark
- it is also robust under stronger stochasticity, higher burden, delayed support, fragile tissue, and movement bottlenecks

## What More Training Means

More training is not automatically better from this point.

What may still help later:

- more careful DAgger data aggregation
- training multiple seeds and selecting by held-out validation
- conservative extensions that preserve the DAgger policy

What is not currently justified:

- longer blind RL fine-tuning
- returning to runtime MCTS
- changing the benchmark to make the learner look better

So the next research phase is:

- freeze the DAgger mainline
- report reproducibility
- report robustness
- finalize paper tables
- only then consider carefully constrained training extensions

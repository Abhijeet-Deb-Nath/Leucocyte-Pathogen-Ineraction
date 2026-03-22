# Reproducibility Summary

## Benchmark

All results in this note refer to the frozen held-out `100`-seed evaluation benchmark.

## Heuristic Reference

- `win_rate = 0.58`
- `tissue_damage = 94.1710`
- `host_utility = -90.3152`
- `episode_length = 162.76`
- `recruited_neutrophils = 2.22`

## Hierarchical BC + DAgger

### Training Seed 42

- `win_rate = 0.60`
- `tissue_damage = 90.6470`
- `host_utility = -86.72115`
- `episode_length = 137.07`
- `recruited_neutrophils = 4.44`

### Training Seed 43

- `win_rate = 0.50`
- `tissue_damage = 94.2635`
- `host_utility = -90.88425`
- `episode_length = 149.09`
- `recruited_neutrophils = 3.11`

### Training Seed 44

- `win_rate = 0.54`
- `tissue_damage = 94.0575`
- `host_utility = -91.20535`
- `episode_length = 145.73`
- `recruited_neutrophils = 3.35`

## Takeaway

The method is not a fragile one-off success, but it is also not fully stable enough to claim uniform superiority over the heuristic.

Best summary:

- competitive learned controller
- stronger tissue-preservation tendency
- clear training-seed variance

This supports a careful claim:

- hierarchical `BC + DAgger` is a credible learned mainline for the benchmark

It does not yet support the strongest claim:

- hierarchical `BC + DAgger` always outperforms the heuristic

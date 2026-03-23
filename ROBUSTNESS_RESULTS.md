# Robustness Results

## Status

Robustness evaluation has now been fully rerun for the frozen hierarchical `BC + DAgger` controller, including the corrected `high_bacteria_stochasticity` suite.

The learned controller remains competitive or better than the heuristic in all currently reported stress settings.

Curated artifacts live in:

- `results/robustness/hier_dagger_main/`

## Suite-by-Suite Summary

### `baseline_reference`

- heuristic:
  - `win_rate = 0.58`
  - `tissue_damage = 94.1710`
  - `host_utility = -90.3152`
- learned:
  - `win_rate = 0.60`
  - `tissue_damage = 90.6470`
  - `host_utility = -86.72115`

Interpretation:

- the frozen learned controller remains slightly stronger than the heuristic on the main benchmark slice

### `high_bacteria_stochasticity`

- heuristic:
  - `win_rate = 0.54`
  - `tissue_damage = 88.8430`
  - `host_utility = -85.46935`
- learned:
  - `win_rate = 0.65`
  - `tissue_damage = 84.0700`
  - `host_utility = -80.20075`

Interpretation:

- the learned controller handles increased bacteria randomness clearly better than the heuristic

### `hotspot_overload`

- heuristic:
  - `win_rate = 0.39`
  - `tissue_damage = 129.7790`
  - `host_utility = -124.04345`
- learned:
  - `win_rate = 0.55`
  - `tissue_damage = 123.2315`
  - `host_utility = -117.78575`

Interpretation:

- the learned controller is substantially stronger when the initial bacterial burden is raised

### `recruitment_delay_stress`

- heuristic:
  - `win_rate = 0.51`
  - `tissue_damage = 92.7105`
  - `host_utility = -88.7734`
- learned:
  - `win_rate = 0.57`
  - `tissue_damage = 92.3840`
  - `host_utility = -88.1073`

Interpretation:

- the learned controller handles delayed reinforcement slightly better

### `fragile_tissue`

- heuristic:
  - `win_rate = 0.50`
  - `tissue_damage = 119.5420`
  - `host_utility = -115.92430`
- learned:
  - `win_rate = 0.51`
  - `tissue_damage = 113.0116`
  - `host_utility = -109.53105`

Interpretation:

- the learned controller is more tissue-preserving when damage becomes more costly

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

- movement constraints reduce both policies, but the learned controller holds up better on win rate and utility

## Note On The Stochasticity Fix

The earlier provisional `high_bacteria_stochasticity` output was invalid because bacteria movement had been using a module-level singleton that captured default stochasticity at import time.

That issue is now fixed in `agents/bacteria_adaptive.py`, and the current robustness bundle contains the corrected rerun.

## Overall Interpretation

The robustness evidence strengthens the main claim:

- the hierarchical `BC + DAgger` controller is not just competitive on the frozen benchmark
- it remains strong under increased pathogen randomness, higher starting burden, delayed neutrophil arrival, fragile tissue costs, and narrowed movement corridors

This supports a stronger paper framing around:

- partial observability
- delayed immune support
- adversarial pathogen pressure
- structured memory-based control

# Robustness Results

## Status

Robustness evaluation has now been run for the frozen hierarchical `BC + DAgger` controller.

The learned controller remains competitive or better than the heuristic in multiple stress settings.

## Reliable Suites To Report

These suites produced meaningful stress differences and can be used in the paper-track discussion:

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

- the learned controller is substantially stronger when initial bacterial burden is raised

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

## Provisional Suite

### `high_bacteria_stochasticity`

This suite matched the baseline exactly in the first robustness run.

That result should not be trusted yet.

Reason:

- bacteria movement used a module-level singleton that captured default stochasticity at import time
- the robustness override was therefore ineffective
- this has now been fixed in `agents/bacteria_adaptive.py`

Action:

- rerun `high_bacteria_stochasticity` before using it in paper tables

## Overall Interpretation

The robustness evidence strengthens the main claim:

- the hierarchical `BC + DAgger` controller is not just competitive on the frozen benchmark
- it is especially promising under pressure-heavy and delay-heavy adversarial settings

This supports a stronger paper framing around:

- partial observability
- delayed immune support
- adversarial pathogen pressure
- structured memory-based control

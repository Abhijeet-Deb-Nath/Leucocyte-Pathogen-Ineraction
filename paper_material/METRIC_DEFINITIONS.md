# Metric Definitions

This file defines the metrics reported in the paper from `experiments/evaluate_policies.py`.

## Outcome Metrics

### `win_rate`

Definition:

- the mean of the binary indicator `win`
- `win = 1` if `winner == "Host"`
- `win = 0` otherwise

Interpretation:

- fraction of episodes in which the host clears infection without entering a losing terminal state

### `tissue_damage`

Definition:

- final accumulated tissue damage at episode end

Interpretation:

- lower is better
- this is a direct damage metric, not normalized by episode length

### `host_utility`

Definition:

- final value of `env.compute_host_utility()` at episode end

Interpretation:

- higher is better
- this is the project's aggregate task objective, combining bacterial clearance benefit with penalties for damage and immune usage

### `episode_length`

Definition:

- total number of environment steps before termination

Interpretation:

- lower can indicate faster containment or faster failure
- therefore it should never be interpreted alone

### `recruited_neutrophils`

Definition:

- total number of neutrophils recruited over the episode

Interpretation:

- not intrinsically good or bad
- a support-usage statistic that must be interpreted jointly with utility and damage

## Behavior Metrics

### `attack_actions`

Definition:

- total count of macrophage attack actions in an episode

### `signal_actions`

Definition:

- total count of signaling actions, aggregated across all concrete signal intensities

### `blind_signal_actions`

Definition:

- signal actions taken when no local bacteria were visible inside macrophage sense radius

Interpretation:

- useful for identifying over-escalation or chemokine-only escalation

### `stay_actions`

Definition:

- total count of no-op movement actions `("move", (0, 0))`

### `coverage_ratio`

Definition:

- number of distinct walkable cells visited by the macrophage divided by total walkable cells

Interpretation:

- coarse exploration footprint over the episode

### `corner_dwell_ratio`

Definition:

- fraction of steps spent at one of the four grid corners

Interpretation:

- diagnostic for camping or degenerate edge behavior

### `visible_bacteria_ratio`

Definition:

- fraction of steps in which bacteria were visible inside macrophage sense radius

Interpretation:

- proxy for time spent in direct contact with local threat

### `camp_signal_signature`

Definition:

- binary diagnostic triggered when:
  - corner dwell is high or exploration is extremely low
  - signal usage is high
  - attack usage is low

Interpretation:

- coarse detector for pathological camp-and-signal behavior

## Aggregation Rule

Per-episode metrics are first recorded for every seed and policy. Summary tables report arithmetic means across episodes for each policy.

This means the paper's default tables are mean-performance summaries, not medians or confidence intervals.

## Writing Guidance

When reporting results:

- use `host_utility` and `tissue_damage` together
- do not interpret `episode_length` alone
- treat `recruited_neutrophils` and `signal_actions` as behavior descriptors, not primary objectives

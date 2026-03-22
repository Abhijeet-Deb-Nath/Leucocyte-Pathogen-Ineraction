# Research Pathway

## Mainline

This project now follows a single paper-track direction:

- benchmark: partially observable stochastic host-pathogen environment
- adversary: adaptive rule-based bacteria
- baseline teacher: heuristic macrophage controller
- learned mainline: hierarchical recurrent host controller
- deployment target: fast one-pass neural inference for GUI

Rejected as mainlines:

- runtime MCTS in the GUI
- NN-guided runtime MCTS in the GUI
- flat PPO-from-scratch
- BC-only as the final deployed method

## Why This Path

The task is a POMDP with:

- local sensing only
- delayed neutrophil recruitment
- stochastic pathogen behavior
- collateral-damage tradeoffs

The learned controller therefore needs:

- memory
- structured decisions
- expert-guided bootstrap
- post-bootstrap improvement on the real task objective

## Method

### Stage A: Sequence-Aware Imitation

Train a recurrent controller on trajectory chunks from the heuristic teacher.

Supervision includes:

- macro mode label
- micro action label
- privileged training-only auxiliary targets

Macro modes:

- `engage`
- `request_help`
- `follow_chem`
- `patrol`

Micro actions:

- move
- attack
- stay

`request_help` is abstract. The environment wrapper resolves it to concrete signal intensity from local pressure.

### Stage B: DAgger

Run the learner in the real environment, relabel learner-visited states with heuristic `(mode, action)` targets, aggregate the dataset, and retrain.

Goal:

- reduce learner-state collapse
- reduce no-signal and over-signal failure modes
- preserve fast inference

### Stage C: Asymmetric RL Fine-Tuning

After BC + DAgger becomes credible:

- actor sees partial observation only
- critic can use privileged full-state training signals
- optimize host utility without changing inference inputs

## Acceptance Gates

### BC Gate

- macro accuracy clearly above chance
- micro accuracy above the old weak plateau
- nonzero attack on held-out seeds
- nonzero contextual escalation on held-out seeds

### DAgger Gate

- better learner-state recovery than BC-only
- reduced collapse modes
- nonzero recruitment on at least some held-out episodes

### RL Fine-Tune Gate

- better host utility than BC + DAgger
- competitive with heuristic on held-out seeds
- still responsive in the GUI on local hardware

## Environment Policy

The benchmark environment is frozen.

Allowed changes:

- bug fixes
- diagnostics
- training curricula that do not alter final evaluation rules

Not allowed:

- weakening the benchmark to make the learner look better
- reintroducing runtime search as the deployed solution

## Repo Intent

The repo should be read as:

- a benchmark and simulation environment
- a heuristic teacher and baseline
- a single learned-controller research path

It should not be read as:

- a collection of competing dead-end agent branches
- a planner-centric demo project
- a heuristic-only showcase

# Project Status

## Current Standing

This project is no longer pretending to have multiple equally alive algorithmic paths.

The repo is now organized around:
- one fixed partial-observation environment
- one heuristic teacher / baseline
- one learned-controller mainline

The learned controller has now reached a credible paper-track state.

Current best learned method:
- hierarchical recurrent `BC + DAgger`

Current frozen checkpoint:
- `models/frozen/hier_dagger_main.pt`

Current result snapshot:
- the frozen DAgger checkpoint is competitive with the heuristic and was slightly better on the main `100`-seed held-out evaluation
- reproducibility across training seeds is mixed but still supports a competitive-method claim
- robustness evaluation is also promising under multiple stress settings
- see `RESULTS_SUMMARY.md`, `ROBUSTNESS_RESULTS.md`, and `results/final_eval/REPRODUCIBILITY_SUMMARY.md`

## What Is In The Code Right Now

- partial-observation recurrent policy inference
- hierarchical macro-mode supervision
- abstract `request_help` action for the learned controller
- sequence-aware recurrent behavior-cloning bootstrap
- DAgger-style learner-state relabeling
- optional RL fine-tuning after the DAgger stage
- heuristic-vs-learned evaluation script
- explicit-policy simulation runner with GUI support

## What Was Removed

Removed from the active codebase:
- runtime MCTS and planner-specific hooks
- hidden planner fallback inside the simulator
- planner-vs-RL comparison scripts
- exploratory bacteria-mode and parameter-sweep scripts tied to the removed default control path
- versioned local CSV artifacts that were muddying the project goal

These were removed because they were no longer the most promising path for:
- partial observability
- stochastic dynamics
- local hardware constraints
- live GUI demonstration

## Honest Assessment

Current evidence says:
- the heuristic is a strong teacher and baseline
- flat PPO was not a good fit
- planner-based control was too expensive and too weak for the demo setting
- sequence-aware BC alone was not enough
- hierarchical `BC + DAgger` produced the first competitive learned controller
- the first RL fine-tuning attempt degraded the DAgger policy

So the project is no longer blocked by “no viable learned path.” The next work is now about consolidation, reproducibility, robustness, and careful extensions.

## Active Research Direction

The repo is now aligned around a single plan:

1. Keep the benchmark environment fixed.
2. Keep the heuristic only as teacher and baseline.
3. Use hierarchical recurrent `BC + DAgger` as the main learned controller.
4. Revisit RL fine-tuning only if it can improve the frozen DAgger checkpoint without destabilizing it.

The next experiment steps should be:
- rerun and validate the `high_bacteria_stochasticity` robustness suite after the bacteria-policy bug fix
- finalize paper tables for `heuristic`, `BC-only`, `BC+DAgger`, and `BC+DAgger+RL`
- organize robustness tables around the now-validated stress suites
- keep GUI validation centered on the frozen DAgger checkpoint

These are now organization-and-evidence steps, not “find a new mainline” steps.

## What The Repo Is For Now

This codebase should now be used for:
- environment development
- heuristic teacher inspection
- learned-controller research on the hierarchical mainline
- remote training/evaluation experiments on the single active mainline
- final demo and paper artifact organization

This codebase should not be treated as:
- a planner benchmark zoo
- a storage location for generated artifacts
- a record of every abandoned algorithmic branch

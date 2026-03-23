# Project Status

## Current Standing

This project is no longer spread across multiple equally alive algorithm branches.

The repo is now organized around:

- one fixed partial-observation environment
- one heuristic teacher / baseline
- one learned-controller mainline

Current best learned method:

- hierarchical recurrent `BC + DAgger`

Current frozen checkpoint:

- `models/frozen/hier_dagger_main.pt`

Current evidence snapshot:

- the full four-row held-out comparison table is now complete
- the frozen DAgger checkpoint slightly beat the heuristic on the main 100-seed held-out benchmark
- BC-only collapsed on the same benchmark, confirming that plain imitation is not enough
- RL fine-tuning remained worse than the frozen DAgger checkpoint
- reproducibility across training seeds is mixed, but supports a competitive-method claim
- the corrected full robustness suite now also supports the frozen learned controller across all reported stress settings

## What Is In The Code Right Now

- partial-observation recurrent policy inference
- hierarchical macro-mode supervision
- abstract `request_help` action for the learned controller
- sequence-aware recurrent behavior-cloning bootstrap
- DAgger-style learner-state relabeling
- optional RL fine-tuning after the DAgger stage
- heuristic-vs-learned evaluation scripts
- robustness suite with temporary config overrides
- explicit-policy simulation runner with GUI support

## What Was Removed

Removed from the active paper path:

- runtime MCTS and planner-specific hooks
- hidden planner fallback inside the simulator
- planner-vs-learned comparison scripts
- exploratory planner-era sweep artifacts
- redundant local result dumps outside curated result folders

These removals were made because they were no longer the strongest path for:

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

So the project is no longer blocked by "no viable learned path." The work now is about evidence, reporting, and careful future extensions.

## Active Research Direction

The repo is now aligned around this plan:

1. Keep the benchmark environment fixed.
2. Keep the heuristic only as teacher and baseline.
3. Use hierarchical recurrent `BC + DAgger` as the main learned controller.
4. Treat RL fine-tuning only as an ablation until it can improve the frozen checkpoint without destabilizing it.

## Next Concrete Work

The next work should be:

- keep the completed main comparison table for `heuristic`, `BC-only`, `BC + DAgger`, and `BC + DAgger + RL` as the fixed paper result
- keep the reproducibility table centered on training seeds `42`, `43`, and `44`
- use the corrected robustness bundle in `results/robustness/hier_dagger_main/`
- prepare paper figures, behavior-difference writeup, and a short GUI demo around the frozen DAgger checkpoint

These are evidence-and-packaging steps, not "find a brand new mainline" steps.

## What The Repo Is For Now

This codebase should now be used for:

- environment development
- heuristic teacher inspection
- learned-controller research on the hierarchical mainline
- remote training/evaluation experiments on the single active mainline
- final demo and paper artifact organization

This codebase should not be treated as:

- a planner benchmark zoo
- a storage location for arbitrary generated artifacts
- a record of every abandoned algorithmic branch

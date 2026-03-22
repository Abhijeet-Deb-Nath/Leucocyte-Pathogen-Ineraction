# Project Status

## Current Standing

This project is no longer pretending to have multiple equally alive algorithmic paths.

The repo is now organized around:
- one fixed partial-observation environment
- one heuristic teacher / baseline
- one learned-controller mainline

The learned controller is **not paper-ready yet**. The environment and teacher are solid, but the learned policy still needs a stronger training formulation.

## What Is In The Code Right Now

- partial-observation recurrent policy inference
- abstract `request_help` action for the learned controller
- sequence-aware recurrent behavior-cloning bootstrap
- optional PPO fine-tuning after the BC stage
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
- the heuristic remains the only clearly competent controller
- flat PPO was not a good fit
- planner-based control was too expensive and too weak for the demo setting
- sequence-aware BC alone is still not enough

So the real blocker is **algorithm design**, not just training time or hardware.

## Active Research Direction

The repo is now aligned around a single plan:

1. Use the heuristic only to bootstrap the learned controller.
2. Improve supervision on the learner's own visited states.
3. Add RL fine-tuning only after imitation becomes behaviorally stable.

The next algorithmic steps should be:
- add high-level mode supervision such as `engage`, `request_help`, `follow_chem`, `patrol`
- add DAgger-style dataset aggregation
- revisit RL fine-tuning only after the policy can reliably imitate useful escalation behavior

## What The Repo Is For Now

This codebase should now be used for:
- environment development
- heuristic teacher inspection
- learned-controller research
- remote training experiments on the single active mainline

This codebase should not be treated as:
- a planner benchmark zoo
- a storage location for generated artifacts
- a record of every abandoned algorithmic branch

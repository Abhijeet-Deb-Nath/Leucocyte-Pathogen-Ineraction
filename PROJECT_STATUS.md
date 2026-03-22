# Project Status

## Current Stage

This project is now in a **clean paper-track transition** stage.

Main direction:
- partial-observation adversarial immune control
- recurrent PPO as the main learned controller

Secondary baselines:
- heuristic local controller
- lightweight MCTS local planner
- feedforward PPO baseline

## What Is Already Done

- partial observation is the mainline benchmark
- shortcut-heavy full-state results have been deprioritized
- local heuristic and MCTS baselines exist
- feedforward PPO baseline exists
- recurrent PPO has been integrated cleanly
- recurrent inference now carries hidden state across steps and resets between episodes
- masked recurrent PPO support is integrated
- invalid-action fallback collapse in recurrent PPO was fixed locally

## Important Local Finding

Before masked recurrent PPO, the tiny recurrent path was structurally bad:
- invalid recurrent actions were being sanitized frequently
- this biased the policy into degenerate behavior

After masked recurrent PPO:
- fallback rate dropped from `71/80` to `0/120` in direct local tracing

That means the recurrent path is now structurally sound enough for remote training.

## What Is Not Finished Yet

- no strong recurrent checkpoint exists yet
- the moderate local recurrent smoke run still showed weak engagement
- policy quality is still below paper-ready standard

So the remaining bottleneck is:
- **training quality and scale**

not:
- repo structure
- recurrent inference wiring
- hidden-state reset correctness

## Current Kept Local Reference

Current kept local baseline snapshot:
- [partial_mainline_best_current_5ep_summary.csv](/C:/Users/Ankon/Desktop/Projects/AI/results/final_eval/partial_mainline_best_current_5ep_summary.csv)

Reference local baseline:
- heuristic: `0.6`
- mcts: `0.4`
- feedforward rl: `0.4`

These are the numbers the recurrent remote smoke run should try to beat.

## Immediate Next Step

Run a **moderate remote recurrent smoke experiment**.

Goal of that run:
- confirm real engagement behavior under more timesteps
- check whether recurrent PPO beats the feedforward baseline

## Acceptance Criteria For The First Remote Recurrent Run

The first remote recurrent smoke run is promising if it shows:
- attack actions are clearly nonzero
- signaling is contextual, not blind
- no stay-collapse or fallback-collapse
- GUI behavior looks purposeful on representative seeds
- 5-seed summary is at least competitive with the current feedforward baseline

## If The Remote Recurrent Smoke Run Fails

Do not scale blindly.

Come back to:
- richer partial-observation memory design
- recurrent behavior cloning quality
- per-seed trace tooling
- reward/engagement tuning

## Repo Cleanup Policy

This repo now keeps:
- source code
- dependency files
- paper-facing documentation
- a very small number of kept summary artifacts

This repo intentionally does not keep:
- temporary smoke checkpoints
- local probe artifacts
- stale full-state artifacts
- old exploratory outputs that are no longer part of the paper path

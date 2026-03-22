"""Evaluate heuristic, MCTS, and RL macrophage policies on shared metrics."""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

import config
from agents.heuristic import HeuristicAgent
from agents.mcts import MCTSAgent
from agents.rl_agent import RLMacrophageAgent
from simulator.environment import Environment


def _build_policy(
    policy_name,
    rl_model_path=None,
    obs_mode="partial_state",
    mcts_rollout_budget=None,
    mcts_rollout_depth=None,
):
    if policy_name == "heuristic":
        return HeuristicAgent()
    if policy_name == "mcts":
        return MCTSAgent(rollout_budget=mcts_rollout_budget, rollout_depth=mcts_rollout_depth)
    if policy_name == "rl":
        if not rl_model_path:
            raise ValueError("rl_model_path is required when policy_name='rl'")
        return RLMacrophageAgent(model_path=rl_model_path, observation_mode=obs_mode)
    raise ValueError(f"Unknown policy: {policy_name}")


def run_policy_eval(
    policy_name,
    episodes,
    base_seed,
    rl_model_path=None,
    obs_mode="partial_state",
    mcts_rollout_budget=None,
    mcts_rollout_depth=None,
    max_steps=None,
):
    policy = _build_policy(
        policy_name,
        rl_model_path=rl_model_path,
        obs_mode=obs_mode,
        mcts_rollout_budget=mcts_rollout_budget,
        mcts_rollout_depth=mcts_rollout_depth,
    )
    records = []

    for ep in range(episodes):
        seed = None if base_seed is None else base_seed + ep
        env = Environment(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset(env)

        action_counts = Counter()
        visited_positions = {env.macrophage.position}
        corner_steps = 0
        visible_bacteria_steps = 0
        blind_signal_actions = 0
        walkable_cells = sum(
            1 for y in range(env.height) for x in range(env.width) if env._is_walkable((x, y))
        )

        while not env.done:
            if max_steps is not None and env.step_count >= max_steps:
                break
            if env.macrophage.position in {
                (0, 0),
                (0, env.height - 1),
                (env.width - 1, 0),
                (env.width - 1, env.height - 1),
            }:
                corner_steps += 1
            if env.get_local_bacteria(env.macrophage.position, config.MACROPHAGE_SENSE_RADIUS):
                visible_bacteria_steps += 1
            action = policy.choose_action(env)
            if action[0] in {"signal", "signal_low", "signal_medium", "signal_high"} and not env.get_local_bacteria(
                env.macrophage.position, config.MACROPHAGE_SENSE_RADIUS
            ):
                blind_signal_actions += 1
            action_counts[action] += 1
            env.step(macrophage_action=action)
            visited_positions.add(env.macrophage.position)

        total_steps = max(1, env.step_count)
        attack_actions = int(action_counts[("attack", None)])
        signal_actions = int(
            action_counts[("signal_low", None)]
            + action_counts[("signal_medium", None)]
            + action_counts[("signal_high", None)]
            + action_counts[("signal", None)]
        )
        stay_actions = int(action_counts[("move", (0, 0))])
        coverage_ratio = float(len(visited_positions) / max(1, walkable_cells))
        corner_dwell_ratio = float(corner_steps / total_steps)
        visible_bacteria_ratio = float(visible_bacteria_steps / total_steps)
        camp_signal_signature = int(
            (corner_dwell_ratio >= 0.35 or coverage_ratio <= 0.02)
            and signal_actions >= 4
            and attack_actions <= 2
        )

        records.append(
            {
                "policy": policy_name,
                "episode": ep,
                "seed": seed,
                "winner": env.winner,
                "win": 1 if env.winner == "Host" else 0,
                "tissue_damage": float(env.tissue_damage),
                "host_utility": float(env.compute_host_utility()),
                "episode_length": int(env.step_count),
                "recruited_neutrophils": int(env.recruited_neutrophils_total),
                "attack_actions": attack_actions,
                "signal_actions": signal_actions,
                "blind_signal_actions": int(blind_signal_actions),
                "stay_actions": stay_actions,
                "coverage_ratio": coverage_ratio,
                "corner_dwell_ratio": corner_dwell_ratio,
                "visible_bacteria_ratio": visible_bacteria_ratio,
                "camp_signal_signature": camp_signal_signature,
                "truncated": 1 if (max_steps is not None and env.step_count >= max_steps and not env.done) else 0,
            }
        )

    return pd.DataFrame(records)


def evaluate_all(
    episodes,
    base_seed,
    rl_model_path,
    obs_mode,
    output_csv,
    mcts_rollout_budget=None,
    mcts_rollout_depth=None,
    max_steps=None,
):
    frames = [
        run_policy_eval(
            "heuristic",
            episodes=episodes,
            base_seed=base_seed,
            max_steps=max_steps,
        ),
        run_policy_eval(
            "mcts",
            episodes=episodes,
            base_seed=base_seed,
            mcts_rollout_budget=mcts_rollout_budget,
            mcts_rollout_depth=mcts_rollout_depth,
            max_steps=max_steps,
        ),
        run_policy_eval(
            "rl",
            episodes=episodes,
            base_seed=base_seed,
            rl_model_path=rl_model_path,
            obs_mode=obs_mode,
            mcts_rollout_budget=mcts_rollout_budget,
            mcts_rollout_depth=mcts_rollout_depth,
            max_steps=max_steps,
        ),
    ]
    df = pd.concat(frames, ignore_index=True)

    summary = (
        df.groupby("policy")
        .agg(
            win_rate=("win", "mean"),
            tissue_damage=("tissue_damage", "mean"),
            host_utility=("host_utility", "mean"),
            episode_length=("episode_length", "mean"),
            recruited_neutrophils=("recruited_neutrophils", "mean"),
            attack_actions=("attack_actions", "mean"),
            signal_actions=("signal_actions", "mean"),
            blind_signal_actions=("blind_signal_actions", "mean"),
            stay_actions=("stay_actions", "mean"),
            coverage_ratio=("coverage_ratio", "mean"),
            corner_dwell_ratio=("corner_dwell_ratio", "mean"),
            visible_bacteria_ratio=("visible_bacteria_ratio", "mean"),
            camp_signal_signature=("camp_signal_signature", "mean"),
        )
        .reset_index()
    )

    df.to_csv(output_csv, index=False)
    summary_path = str(Path(output_csv).with_name(Path(output_csv).stem + "_summary.csv"))
    summary.to_csv(summary_path, index=False)

    print("Per-episode results:")
    print(df.head(min(12, len(df))).to_string(index=False))
    print("\nSummary metrics:")
    print(summary.to_string(index=False))
    print(f"\nSaved detailed results to: {output_csv}")
    print(f"Saved summary results to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate heuristic vs MCTS vs RL policies")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per policy")
    parser.add_argument("--seed", type=int, default=123, help="Base seed for reproducible evaluation")
    parser.add_argument(
        "--rl-model",
        type=str,
        required=True,
        help="Path to saved PPO model (zip path or stable-baselines base name)",
    )
    parser.add_argument(
        "--obs-mode",
        type=str,
        default="partial_state",
        choices=["full_state", "partial_state"],
        help="Observation mode expected by the RL model",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="policy_eval_results.csv",
        help="Where to save per-episode policy comparison results",
    )
    parser.add_argument(
        "--mcts-rollout-budget",
        type=int,
        default=None,
        help="Optional MCTS rollout budget override for evaluation speed control",
    )
    parser.add_argument(
        "--mcts-rollout-depth",
        type=int,
        default=None,
        help="Optional MCTS rollout depth override for evaluation speed control",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional hard cap on steps per episode for quick evaluations",
    )

    args = parser.parse_args()
    evaluate_all(
        episodes=max(1, args.episodes),
        base_seed=args.seed,
        rl_model_path=args.rl_model,
        obs_mode=args.obs_mode,
        output_csv=args.output_csv,
        mcts_rollout_budget=args.mcts_rollout_budget,
        mcts_rollout_depth=args.mcts_rollout_depth,
        max_steps=args.max_steps,
    )

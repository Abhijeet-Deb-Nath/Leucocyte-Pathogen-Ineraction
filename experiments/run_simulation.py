"""Run a single Pathway 1 simulation and save outputs."""

import argparse

import pandas as pd

from agents.heuristic import HeuristicAgent
from agents.mcts import MCTSAgent
from agents.rl_agent import RLMacrophageAgent
from simulator.environment import Environment


def _build_policy(policy, rl_model_path=None, rl_obs_mode="partial_state"):
    if policy == "heuristic":
        return HeuristicAgent()
    if policy == "mcts":
        return MCTSAgent()
    if policy == "rl":
        if not rl_model_path:
            raise ValueError("--rl-model-path is required when --policy rl is selected")
        return RLMacrophageAgent(model_path=rl_model_path, observation_mode=rl_obs_mode)
    raise ValueError(f"Unknown policy mode: {policy}")


def run(headless=True, verbose=False, policy="mcts", rl_model_path=None, rl_obs_mode="partial_state"):
    env = Environment()
    policy_agent = _build_policy(policy, rl_model_path=rl_model_path, rl_obs_mode=rl_obs_mode)
    if hasattr(policy_agent, "reset"):
        policy_agent.reset(env)

    if not headless:
        from visualization.pyqt_gui import launch_gui

        launch_gui(env, policy_agent=policy_agent)
    else:
        while not env.done:
            action = policy_agent.choose_action(env)
            env.step(macrophage_action=action)

    save_results(env)
    if verbose:
        print_summary(env)


def save_results(env):
    df = pd.DataFrame(env.history)
    df.to_csv("simulation_history_pathway1.csv", index=False)


def print_summary(env):
    print("=" * 64)
    print("PATHWAY 1 SIMULATION SUMMARY")
    print("=" * 64)
    print(f"Winner: {env.winner}")
    print(f"Reason: {env.win_reason}")
    print(f"Steps: {env.step_count}")
    print(f"Final Macrophage Health: {env.macrophage.health}")
    print(f"Final Bacteria Count: {len(env.bacteria)}")
    print(f"Final Neutrophil Count: {len(env.neutrophils)}")
    print(f"Final Tissue Damage: {env.tissue_damage:.2f}")
    print(f"Host Utility: {env.compute_host_utility():.2f}")
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one Pathway 1 simulation")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print terminal summary after the run",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="mcts",
        choices=["heuristic", "mcts", "rl"],
        help="Macrophage policy mode",
    )
    parser.add_argument(
        "--rl-model-path",
        type=str,
        default=None,
        help="Path to PPO model used when --policy rl",
    )
    parser.add_argument(
        "--rl-obs-mode",
        type=str,
        default="partial_state",
        choices=["full_state", "partial_state"],
        help="Observation mode expected by RL model",
    )
    args = parser.parse_args()
    run(
        headless=args.headless,
        verbose=args.verbose,
        policy=args.policy,
        rl_model_path=args.rl_model_path,
        rl_obs_mode=args.rl_obs_mode,
    )

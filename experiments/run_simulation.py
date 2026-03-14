"""Run a single Pathway 1 simulation and save outputs."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from simulator.environment import Environment


def run(headless=True, verbose=False):
    env = Environment()

    if not headless:
        from visualization.pyqt_gui import launch_gui

        launch_gui(env)
    else:
        while not env.done:
            env.step()

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
    args = parser.parse_args()
    run(headless=args.headless, verbose=args.verbose)

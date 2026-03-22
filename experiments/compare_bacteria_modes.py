"""Compare Pathway 1 bacteria policy settings by adaptive stochasticity."""

import matplotlib.pyplot as plt
import pandas as pd

import config
from simulator.environment import Environment


def run_comparison_experiments(trials_per_mode=8):
    settings = [
        ("adaptive_low_noise", 0.05),
        ("adaptive_default", 0.12),
        ("adaptive_high_noise", 0.25),
    ]

    original_noise = config.BACTERIA_STOCHASTICITY
    results = []

    try:
        for name, noise in settings:
            config.BACTERIA_STOCHASTICITY = noise
            wins = 0
            damages = []
            utilities = []
            durations = []

            for trial in range(trials_per_mode):
                env = Environment(seed=trial)
                while not env.done:
                    env.step()

                if env.winner == "Host":
                    wins += 1
                damages.append(env.tissue_damage)
                utilities.append(env.compute_host_utility())
                durations.append(env.step_count)

            results.append(
                {
                    "setting": name,
                    "stochasticity": noise,
                    "host_win_rate": wins / trials_per_mode,
                    "avg_tissue_damage": sum(damages) / len(damages),
                    "avg_host_utility": sum(utilities) / len(utilities),
                    "avg_duration": sum(durations) / len(durations),
                }
            )
    finally:
        config.BACTERIA_STOCHASTICITY = original_noise

    return results


def plot_comparison_results(results):
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].bar(df["setting"], df["host_win_rate"], color="#1976d2")
    axes[0].set_title("Host Win Rate")
    axes[0].set_ylim(0, 1)

    axes[1].bar(df["setting"], df["avg_tissue_damage"], color="#ef6c00")
    axes[1].set_title("Average Tissue Damage")

    axes[2].bar(df["setting"], df["avg_host_utility"], color="#6a1b9a")
    axes[2].set_title("Average Host Utility")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig("pathway1_bacteria_policy_comparison.png", dpi=150)
    plt.close(fig)


def main():
    results = run_comparison_experiments(trials_per_mode=8)
    plot_comparison_results(results)

    df = pd.DataFrame(results)
    df.to_csv("pathway1_bacteria_policy_comparison.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

"""Pathway 1 parameter sweep focused on recruitment delay and damage scaling."""

import pandas as pd

import config
from simulator.environment import Environment


def run_param_sweep():
    original_delay = config.NEUTROPHIL_ARRIVAL_DELAY
    original_damage = config.TISSUE_DAMAGE_FROM_NEUTROPHILS

    scenarios = [
        (5, 0.35),
        (7, 0.45),
        (10, 0.55),
    ]

    records = []
    try:
        for delay, damage in scenarios:
            config.NEUTROPHIL_ARRIVAL_DELAY = delay
            config.TISSUE_DAMAGE_FROM_NEUTROPHILS = damage

            for trial in range(6):
                env = Environment(seed=trial)
                while not env.done:
                    env.step()

                records.append(
                    {
                        "delay": delay,
                        "damage_per_neutrophil": damage,
                        "trial": trial,
                        "winner": env.winner,
                        "steps": env.step_count,
                        "tissue_damage": env.tissue_damage,
                        "host_utility": env.compute_host_utility(),
                        "final_bacteria": len(env.bacteria),
                    }
                )
                print(
                    f"delay={delay}, damage={damage}, trial={trial}, "
                    f"winner={env.winner}, utility={env.compute_host_utility():.2f}"
                )
    finally:
        config.NEUTROPHIL_ARRIVAL_DELAY = original_delay
        config.TISSUE_DAMAGE_FROM_NEUTROPHILS = original_damage

    df = pd.DataFrame(records)
    df.to_csv("pathway1_param_sweep_results.csv", index=False)
    return df


if __name__ == "__main__":
    summary = run_param_sweep()
    print(summary.groupby(["delay", "damage_per_neutrophil"])["host_utility"].mean())

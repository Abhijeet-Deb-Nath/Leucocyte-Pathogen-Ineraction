"""
Example of parameter sweep experiments.
This script runs multiple simulations with different parameters and logs the outcomes.
"""
import pandas as pd
from simulator.environment import Environment
import config

def run_param_sweep():
    results = []
    # Save original config values that we might modify
    orig_mode = config.BACTERIA_MODE
    orig_vis = config.ENABLE_VISUALIZATION
    orig_snap = config.SAVE_SNAPSHOTS
    orig_plot = config.SAVE_PLOTS
    try:
        # Turn off visualization and saving during sweeps for performance
        config.ENABLE_VISUALIZATION = False
        config.SAVE_SNAPSHOTS = False
        config.SAVE_PLOTS = False
        # Sweep over different bacteria behavior modes as an example
        for mode in ["cluster", "scatter", "replicate", "defend"]:
            config.BACTERIA_MODE = mode
            env = Environment()
            # Run until done or horizon
            for step in range(config.TIME_HORIZON):
                env.step()
                if env.done:
                    break
            # Record result
            outcome = {
                "bacteria_mode": mode,
                "winner": env.winner if env.winner else "None",
                "win_reason": env.win_reason if env.win_reason else "N/A",
                "steps_taken": env.step_count
            }
            results.append(outcome)
            print(f"Mode {mode}: winner={outcome['winner']} (reason: {outcome['win_reason']}) in {outcome['steps_taken']} steps.")
        # Convert results to DataFrame and save
        df = pd.DataFrame(results)
        df.to_csv("param_sweep_results.csv", index=False)
    finally:
        # Restore original config
        config.BACTERIA_MODE = orig_mode
        config.ENABLE_VISUALIZATION = orig_vis
        config.SAVE_SNAPSHOTS = orig_snap
        config.SAVE_PLOTS = orig_plot

if __name__ == "__main__":
    run_param_sweep()

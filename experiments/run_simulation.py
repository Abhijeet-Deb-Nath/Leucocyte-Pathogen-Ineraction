"""
Run a single simulation with PyQt5 GUI and save results.
"""
import sys
from pathlib import Path
# Add parent directory to path so we can import from simulator, agents, etc.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from simulator.environment import Environment
from visualization.pyqt_gui import launch_gui
from visualization.plots import save_performance_plots, save_heatmap
import config


def run():
    """Run a single simulation with PyQt5 visualization and save results."""
    # Initialize environment
    print("Initializing simulation environment...")
    env = Environment()
    
    # Launch PyQt5 GUI (blocks until GUI is closed)
    print("Launching PyQt5 GUI...")
    launch_gui(env)
    
    # After GUI closes, save results
    print("\nSaving results...")
    save_results(env)
    print("Simulation complete!")


def save_results(env):
    """Save simulation results to files."""
    # Save performance plots and heatmaps
    if config.SAVE_PLOTS:
        print("  - Generating performance plots...")
        save_performance_plots(env, filename_prefix="performance")
        
        print("  - Generating heatmaps...")
        save_heatmap(env.macrophage_position_frequency, 
                    "Macrophage Position Frequency", 
                    "heatmap_macrophage.png")
        save_heatmap(env.toxin_effect_frequency, 
                    "Toxin Usage Frequency", 
                    "heatmap_toxin.png")
    
    # Save history data to CSV
    print("  - Saving history to CSV...")
    df = pd.DataFrame(env.history)
    df.to_csv("simulation_history.csv", index=False)
    
    # Print summary
    print_summary(env)


def print_summary(env):
    """Print a summary of the simulation outcome."""
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Total Steps: {env.step_count}")
    print(f"Final Macrophage Health: {env.macrophage.health}/100")
    print(f"Final Bacteria Count: {len(env.bacteria)}")
    print(f"Remaining Nutrients: {len(env.nutrients)}")
    
    if env.winner:
        print(f"\n🏆 Winner: {env.winner}")
        print(f"   Reason: {env.win_reason}")
    else:
        print(f"\n⚠️  No decisive winner (simulation timeout)")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    run()

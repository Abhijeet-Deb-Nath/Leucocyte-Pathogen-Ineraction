"""
Compare adaptive bacteria AI vs traditional fixed behavior modes.
Runs simulations with different bacteria strategies and measures performance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from simulator.environment import Environment
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_comparison_experiments(trials_per_mode=10):
    """
    Compare different bacteria behavior strategies.
    
    Tests:
    1. Adaptive AI (context-based with stochasticity)
    2. Fixed modes: scatter, cluster, defend, replicate
    """
    
    modes_to_test = [
        ("Adaptive", True, None),  # (name, use_adaptive, fixed_mode)
        ("Scatter", False, "scatter"),
        ("Cluster", False, "cluster"),
        ("Defend", False, "defend"),
        ("Replicate", False, "replicate"),
    ]
    
    results = []
    
    print("="*60)
    print("BACTERIA BEHAVIOR COMPARISON EXPERIMENT")
    print("="*60)
    print(f"Trials per mode: {trials_per_mode}")
    print(f"MCTS Budget: {config.ROLLOUT_BUDGET}")
    print(f"Grid Size: {config.GRID_SIZE}x{config.GRID_SIZE}")
    print(f"Initial Bacteria: {config.INITIAL_BACTERIA_COUNT}")
    print("="*60 + "\n")
    
    for mode_name, use_adaptive, fixed_mode in modes_to_test:
        print(f"\n{'='*60}")
        print(f"Testing: {mode_name}")
        print(f"{'='*60}")
        
        mode_results = {
            'mode': mode_name,
            'wins': 0,
            'losses': 0,
            'avg_steps': 0,
            'avg_final_bacteria': 0,
            'avg_final_health': 0,
            'max_bacteria_reached': 0,
        }
        
        for trial in range(trials_per_mode):
            # Configure simulation
            config.BACTERIA_ADAPTIVE_ENABLED = use_adaptive
            if not use_adaptive:
                config.BACTERIA_MODE = fixed_mode
            
            # Run simulation
            env = Environment()
            
            while not env.done:
                env.step()
            
            # Record results
            if env.winner == "Macrophage":
                mode_results['wins'] += 1
            else:
                mode_results['losses'] += 1
            
            mode_results['avg_steps'] += env.step_count
            mode_results['avg_final_bacteria'] += len(env.bacteria)
            mode_results['avg_final_health'] += env.macrophage.health
            mode_results['max_bacteria_reached'] = max(
                mode_results['max_bacteria_reached'],
                max(env.history['bacteria_count'])
            )
            
            print(f"  Trial {trial+1}/{trials_per_mode}: "
                  f"Winner={env.winner}, Steps={env.step_count}, "
                  f"FinalBacteria={len(env.bacteria)}, "
                  f"MacHealth={env.macrophage.health}")
        
        # Calculate averages
        mode_results['avg_steps'] /= trials_per_mode
        mode_results['avg_final_bacteria'] /= trials_per_mode
        mode_results['avg_final_health'] /= trials_per_mode
        mode_results['win_rate'] = mode_results['wins'] / trials_per_mode
        
        results.append(mode_results)
        
        print(f"\n  Summary for {mode_name}:")
        print(f"    Win Rate: {mode_results['win_rate']*100:.1f}%")
        print(f"    Avg Steps: {mode_results['avg_steps']:.1f}")
        print(f"    Avg Final Bacteria: {mode_results['avg_final_bacteria']:.2f}")
        print(f"    Avg Final Mac Health: {mode_results['avg_final_health']:.1f}")
        print(f"    Max Bacteria Reached: {mode_results['max_bacteria_reached']}")
    
    return results


def plot_comparison_results(results):
    """Generate comparison plots."""
    df = pd.DataFrame(results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Bacteria Behavior Strategy Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Win Rate
    ax1 = axes[0, 0]
    colors = ['green' if 'Adaptive' in mode else 'gray' for mode in df['mode']]
    ax1.bar(df['mode'], df['win_rate'] * 100, color=colors, alpha=0.7)
    ax1.set_ylabel('Win Rate (%)', fontsize=12)
    ax1.set_title('Macrophage Win Rate by Bacteria Strategy')
    ax1.set_ylim(0, 100)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Average Steps to Completion
    ax2 = axes[0, 1]
    colors = ['blue' if 'Adaptive' in mode else 'gray' for mode in df['mode']]
    ax2.bar(df['mode'], df['avg_steps'], color=colors, alpha=0.7)
    ax2.set_ylabel('Average Steps', fontsize=12)
    ax2.set_title('Simulation Length (Higher = More Competitive)')
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Average Final Bacteria Count
    ax3 = axes[1, 0]
    colors = ['purple' if 'Adaptive' in mode else 'gray' for mode in df['mode']]
    ax3.bar(df['mode'], df['avg_final_bacteria'], color=colors, alpha=0.7)
    ax3.set_ylabel('Avg Final Bacteria Count', fontsize=12)
    ax3.set_title('Bacteria Survival (Higher = Stronger Defense)')
    ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Max Bacteria Population Reached
    ax4 = axes[1, 1]
    colors = ['orange' if 'Adaptive' in mode else 'gray' for mode in df['mode']]
    ax4.bar(df['mode'], df['max_bacteria_reached'], color=colors, alpha=0.7)
    ax4.set_ylabel('Max Bacteria Population', fontsize=12)
    ax4.set_title('Peak Bacteria Count (Growth Effectiveness)')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('bacteria_strategy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot: bacteria_strategy_comparison.png")
    
    return fig


def print_summary_table(results):
    """Print formatted results table."""
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Identify best performers
    best_bacteria_mode = df.loc[df['win_rate'].idxmin(), 'mode']
    best_challenge = df.loc[df['avg_steps'].idxmax(), 'mode']
    
    print(f"\n🏆 INSIGHTS:")
    print(f"  • Strongest bacteria strategy: {best_bacteria_mode} "
          f"(lowest macrophage win rate: {df['win_rate'].min()*100:.1f}%)")
    print(f"  • Most competitive matchup: {best_challenge} "
          f"(longest avg game: {df['avg_steps'].max():.1f} steps)")
    
    # Check if adaptive is best
    adaptive_row = df[df['mode'] == 'Adaptive'].iloc[0]
    print(f"\n📊 Adaptive Bacteria Performance:")
    print(f"  • Win rate: {adaptive_row['win_rate']*100:.1f}%")
    print(f"  • Avg steps: {adaptive_row['avg_steps']:.1f}")
    print(f"  • Final bacteria: {adaptive_row['avg_final_bacteria']:.2f}")
    
    if adaptive_row['win_rate'] < df['win_rate'].mean():
        print("  ✅ Adaptive strategy is MORE CHALLENGING than average!")
    else:
        print("  ⚠️ Adaptive strategy is less effective than some fixed modes")


def main():
    """Run comparison experiment and generate report."""
    
    # Save original config
    original_adaptive = config.BACTERIA_ADAPTIVE_ENABLED
    original_mode = config.BACTERIA_MODE
    
    try:
        # Run experiments
        results = run_comparison_experiments(trials_per_mode=10)
        
        # Generate visualizations
        plot_comparison_results(results)
        
        # Print summary
        print_summary_table(results)
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv('bacteria_strategy_comparison.csv', index=False)
        print(f"\n✅ Saved results: bacteria_strategy_comparison.csv")
        
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETE!")
        print("="*80)
        
    finally:
        # Restore original config
        config.BACTERIA_ADAPTIVE_ENABLED = original_adaptive
        config.BACTERIA_MODE = original_mode


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import numpy as np

def save_performance_plots(env, filename_prefix="performance"):
    """
    Save performance plots such as bacteria count over time and macrophage health over time.
    Creates two subplots: bacteria count vs steps, and macrophage health vs steps.
    """
    history = env.history
    steps = history["step"]
    bacteria_counts = history["bacteria_count"]
    mac_health = history["macrophage_health"]
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    # Plot bacteria count over time
    axes[0].plot(steps, bacteria_counts, color='red')
    axes[0].set_title("Bacteria Count Over Time")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Bacteria Count")
    # Plot macrophage health over time
    axes[1].plot(steps, mac_health, color='blue')
    axes[1].set_title("Macrophage Health Over Time")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Macrophage Health")
    axes[1].set_ylim(0, max(mac_health) * 1.1)
    fig.tight_layout()
    fig.savefig(f"{filename_prefix}.png")
    plt.close(fig)

def save_heatmap(data, title, filename):
    """
    Save a heatmap of given 2D data (numpy array) with a colorbar.
    """
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(data, cmap='hot', origin='lower')
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Frequency")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

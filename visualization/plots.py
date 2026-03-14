import matplotlib.pyplot as plt


def save_performance_plots(env, filename_prefix="performance"):
    history = env.history
    steps = history["step"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    axes[0, 0].plot(steps, history["bacteria_count"], color="#d32f2f")
    axes[0, 0].plot(steps, history["neutrophil_count"], color="#1976d2")
    axes[0, 0].set_title("Population Dynamics")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].legend(["Bacteria", "Neutrophils"])

    axes[0, 1].plot(steps, history["macrophage_health"], color="#388e3c")
    axes[0, 1].set_title("Macrophage Health")
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Health")

    axes[1, 0].plot(steps, history["tissue_damage"], color="#f57c00")
    axes[1, 0].set_title("Tissue Damage")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Damage")

    axes[1, 1].plot(steps, history["host_utility"], color="#6a1b9a")
    axes[1, 1].set_title("Host Utility")
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("Utility")

    fig.tight_layout()
    fig.savefig(f"{filename_prefix}.png", dpi=150)
    plt.close(fig)


def save_heatmap(data, title, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(data, cmap="magma", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Frequency")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

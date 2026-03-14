import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


class GridVisualizer:
    """Simple matplotlib visualizer for Pathway 1 grid semantics."""

    def __init__(self, env):
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_title("Pathway 1: Acute Alveolar Hotspot")
        self.cmap = ListedColormap([
            "#f3f5f7",  # empty
            "#1e88e5",  # macrophage
            "#d32f2f",  # bacteria
            "#43a047",  # nutrient-rich
            "#303030",  # blocked walls
            "#26c6da",  # neutrophils
        ])
        self.im = self.ax.imshow(env.get_grid(), origin="lower", cmap=self.cmap, vmin=0, vmax=5)
        plt.tight_layout()
        plt.show(block=False)

    def update(self, env):
        self.im.set_data(env.get_grid())
        self.ax.set_title(
            f"Step {env.step_count} | B={len(env.bacteria)} N={len(env.neutrophils)} "
            f"Damage={env.tissue_damage:.1f}"
        )
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_image(self, filename):
        self.fig.savefig(filename, dpi=150, bbox_inches="tight")

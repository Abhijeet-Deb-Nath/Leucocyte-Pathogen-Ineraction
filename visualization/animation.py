import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np

class GridVisualizer:
    """Enhanced utility for visualizing the grid state of the simulation with rich graphics."""
    def __init__(self, env):
        # Setup figure with grid and info panel
        self.fig = plt.figure(figsize=(14, 8))
        self.ax_grid = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=3)
        self.ax_info = plt.subplot2grid((3, 3), (0, 2), rowspan=3)
        
        self.im = None
        self.artists = []  # Store artists for animations
        self.toxin_effects = []  # Store toxin blast effects
        self.movement_trail = []  # Store recent macrophage positions
        self._initialize(env)
        plt.tight_layout()
        plt.show(block=False)  # Show non-blocking for live updates

    def _initialize(self, env):
        # Create gradient background for the grid
        grid = np.zeros((env.size, env.size, 3))  # RGB grid
        grid[:, :] = [0.95, 0.95, 0.98]  # Light blue-gray background
        
        self.im = self.ax_grid.imshow(grid, origin='lower', interpolation='bilinear')
        self.ax_grid.set_title("🦠 Macrophage vs Bacteria Battle 💉", fontsize=14, fontweight='bold')
        self.ax_grid.set_xlabel("X Position", fontsize=10)
        self.ax_grid.set_ylabel("Y Position", fontsize=10)
        
        # Add grid lines for clarity
        self.ax_grid.set_xticks(np.arange(-0.5, env.size, 1), minor=True)
        self.ax_grid.set_yticks(np.arange(-0.5, env.size, 1), minor=True)
        self.ax_grid.grid(which='minor', color='gray', linestyle=':', linewidth=0.5, alpha=0.3)
        self.ax_grid.tick_params(which='minor', size=0)
        
        # Info panel setup
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(0.1, 0.95, '', fontsize=9, verticalalignment='top',
                                           family='monospace')

    def update(self, env):
        """Update the grid visualization with enhanced graphics."""
        # Create base grid with gradient
        grid = np.zeros((env.size, env.size, 3))
        grid[:, :] = [0.95, 0.95, 0.98]  # Light background
        
        # Draw nutrients with glow effect
        for (nx, ny) in env.nutrients:
            grid[ny, nx] = [0.2, 0.9, 0.3]  # Bright green
            # Add glow around nutrient
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    gx, gy = nx + dx, ny + dy
                    if 0 <= gx < env.size and 0 <= gy < env.size and (dx != 0 or dy != 0):
                        grid[gy, gx] = grid[gy, gx] * 0.7 + np.array([0.2, 0.9, 0.3]) * 0.3
        
        # Draw bacteria with health-based coloring
        for b in env.bacteria:
            x, y = b.position
            health_ratio = b.health / 20.0  # Normalized health
            # Red gradient based on health (darker = weaker)
            grid[y, x] = [0.5 + health_ratio * 0.5, 0.1, 0.1]
        
        # Draw macrophage with blue color and vision radius
        mx, my = env.macrophage.position
        grid[my, mx] = [0.1, 0.4, 0.9]  # Bright blue
        
        # Update movement trail
        self.movement_trail.append((mx, my))
        if len(self.movement_trail) > 15:
            self.movement_trail.pop(0)
        
        # Draw movement trail with fading effect
        for i, (tx, ty) in enumerate(self.movement_trail[:-1]):
            alpha = (i + 1) / len(self.movement_trail) * 0.3
            grid[ty, tx] = grid[ty, tx] * (1 - alpha) + np.array([0.1, 0.4, 0.9]) * alpha
        
        self.im.set_data(grid)
        
        # Clear previous artists
        for artist in self.artists:
            artist.remove()
        self.artists = []
        
        # Draw macrophage vision radius circle
        vision_circle = patches.Circle((mx, my), 6, fill=False, edgecolor='cyan', 
                                       linewidth=1.5, linestyle='--', alpha=0.4)
        self.ax_grid.add_patch(vision_circle)
        self.artists.append(vision_circle)
        
        # Draw toxin radius indicator (smaller, around macrophage)
        toxin_circle = patches.Circle((mx, my), 2, fill=False, edgecolor='yellow', 
                                      linewidth=2, linestyle='-', alpha=0.5)
        self.ax_grid.add_patch(toxin_circle)
        self.artists.append(toxin_circle)
        
        # Update info panel
        info_str = f"""
╔════════════════════════╗
║   BATTLE STATISTICS    ║
╚════════════════════════╝

Step: {env.step_count}

🔵 MACROPHAGE
  Health: {env.macrophage.health}/100
  {'█' * (env.macrophage.health // 5)}{'░' * (20 - env.macrophage.health // 5)}
  Position: ({mx}, {my})

🔴 BACTERIA
  Count: {len(env.bacteria)}
  {'█' * min(20, len(env.bacteria))}{'░' * max(0, 20 - len(env.bacteria))}
  
🟢 NUTRIENTS
  Available: {len(env.nutrients)}

{'⚠️  CRITICAL!' if env.macrophage.health < 30 else ''}
{'🎯 FEW BACTERIA!' if len(env.bacteria) < 3 else ''}
{'☠️  OUTBREAK!' if len(env.bacteria) > 30 else ''}

Status: {'ENDED' if env.done else 'RUNNING'}
"""
        if env.done:
            info_str += f"\n{'='*24}\n🏆 {env.winner} WINS!\n{env.win_reason}\n{'='*24}"
        
        self.info_text.set_text(info_str)
        
        # Update title with status
        status_emoji = "⚔️" if not env.done else "🏁"
        self.ax_grid.set_title(
            f"{status_emoji} Step {env.step_count} | Bacteria: {len(env.bacteria)} | Health: {env.macrophage.health}",
            fontsize=12, fontweight='bold'
        )
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save_image(self, filename):
        """Save the current grid visualization to an image file."""
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')

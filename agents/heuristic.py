import random
import config

class HeuristicAgent:
    """Greedy heuristic agent for Macrophage decisions."""
    def choose_action(self, env):
        mx, my = env.macrophage.position
        vision = config.MACROPHAGE_VISION_RADIUS
        # Find bacteria within vision radius
        visible_bacteria = []
        for b in env.bacteria:
            bx, by = b.position
            # Manhattan distance for visibility (could use Euclidean as needed)
            if abs(bx - mx) + abs(by - my) <= vision:
                visible_bacteria.append(b)
        if visible_bacteria:
            # Count bacteria within toxin radius (threat in close range)
            close_targets = 0
            # Count bacteria in immediate danger zone (within 3 cells)
            nearby_threats = 0
            for b in visible_bacteria:
                bx, by = b.position
                dist = abs(bx - mx) + abs(by - my)
                if dist <= config.MACROPHAGE_TOXIN_RADIUS:
                    close_targets += 1
                if dist <= 3:  # Immediate danger zone
                    nearby_threats += 1
            # Decide to use toxin if multiple targets are very close
            if close_targets >= 3:
                return ("toxin", None)
            # If Macrophage health is low or heavily outnumbered by NEARBY bacteria, retreat
            if env.macrophage.health < 0.3 * config.MACROPHAGE_HEALTH or nearby_threats > 5:
                # Retreat: move away from nearest visible bacteria
                nearest = min(visible_bacteria, key=lambda b: abs(b.position[0]-mx) + abs(b.position[1]-my))
                bx, by = nearest.position
                dx = 0
                dy = 0
                if mx < bx:
                    dx = -1
                elif mx > bx:
                    dx = 1
                if my < by:
                    dy = -1
                elif my > by:
                    dy = 1
                # Adjust if the move goes out of bounds
                if mx + dx < 0 or mx + dx >= env.size:
                    dx = 0
                if my + dy < 0 or my + dy >= env.size:
                    dy = 0
                # If no move (cornered), and toxin is available, use toxin as last resort
                if dx == 0 and dy == 0:
                    return ("toxin", None)
                return ("move", (dx, dy))
            else:
                # Pursue: move toward the nearest visible bacteria
                nearest = min(visible_bacteria, key=lambda b: abs(b.position[0]-mx) + abs(b.position[1]-my))
                bx, by = nearest.position
                dx = 0
                dy = 0
                if mx < bx:
                    dx = 1
                elif mx > bx:
                    dx = -1
                if my < by:
                    dy = 1
                elif my > by:
                    dy = -1
                # Adjust if out of bounds (shouldn't happen when moving toward known target in bounds, but just in case)
                if mx + dx < 0 or mx + dx >= env.size:
                    dx = 0
                if my + dy < 0 or my + dy >= env.size:
                    dy = 0
                return ("move", (dx, dy))
        else:
            # No visible bacteria
            if env.nutrients:
                # Move toward nearest nutrient to remove it (deny bacteria)
                nearest_nutrient = min(env.nutrients, key=lambda pos: abs(pos[0]-mx) + abs(pos[1]-my))
                nx, ny = nearest_nutrient
                dx = 0
                dy = 0
                if mx < nx:
                    dx = 1
                elif mx > nx:
                    dx = -1
                if my < ny:
                    dy = 1
                elif my > ny:
                    dy = -1
                # Adjust for bounds
                if mx + dx < 0 or mx + dx >= env.size:
                    dx = 0
                if my + dy < 0 or my + dy >= env.size:
                    dy = 0
                return ("move", (dx, dy))
            # If no nutrients or decided not to target them, move randomly to explore
            dirs = [(1,0), (-1,0), (0,1), (0,-1)]
            # Filter out moves that are out of bounds
            valid_dirs = [(dx,dy) for (dx,dy) in dirs if 0 <= mx+dx < env.size and 0 <= my+dy < env.size]
            if not valid_dirs:
                valid_dirs = [(0,0)]
            dx, dy = random.choice(valid_dirs)
            return ("move", (dx, dy))

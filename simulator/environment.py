import numpy as np
import random
from dataclasses import dataclass
import config
from simulator.entities import Macrophage, Bacteria

class Environment:
    """
    The simulation environment for the macrophage vs Pseudomonas aeruginosa adversarial scenario.
    Manages the grid, entity positions, and rules for each time step.
    """
    def __init__(self):
        # Grid size from config
        self.size = config.GRID_SIZE
        # Initialize Macrophage at center of grid (or near center)
        start_x = start_y = self.size // 2
        self.macrophage = Macrophage(position=(start_x, start_y), health=config.MACROPHAGE_HEALTH)
        # Place initial bacteria in random distinct positions not equal to macrophage's position
        self.bacteria = []
        placed = set([ (start_x, start_y) ])
        for i in range(config.INITIAL_BACTERIA_COUNT):
            # find a random empty position
            while True:
                bx = random.randrange(self.size)
                by = random.randrange(self.size)
                if (bx, by) not in placed:
                    placed.add((bx, by))
                    # Bacteria start at full health (mature)
                    b = Bacteria(position=(bx, by), health=config.BACTERIA_MAX_HEALTH)
                    self.bacteria.append(b)
                    break
        # Place nutrients if enabled
        self.nutrients = set()
        if config.NUTRIENTS_ENABLED:
            for i in range(config.N_INITIAL_NUTRIENTS):
                while True:
                    nx = random.randrange(self.size)
                    ny = random.randrange(self.size)
                    if (nx, ny) not in placed:
                        placed.add((nx, ny))
                        self.nutrients.add((nx, ny))
                        break
        # Simulation state
        self.step_count = 0
        self.done = False
        self.winner = None
        self.win_reason = None
        # Logging histories for metrics
        self.history = {
            "step": [],
            "macrophage_health": [],
            "bacteria_count": [],
            "nutrients_count": []
        }
        # Heatmap data
        self.macrophage_position_frequency = np.zeros((self.size, self.size), dtype=int)
        self.toxin_effect_frequency = np.zeros((self.size, self.size), dtype=int)
        # Record initial state in history
        self._record_history()
        # Mark initial Macrophage position in frequency map
        mx, my = self.macrophage.position
        self.macrophage_position_frequency[my, mx] += 1

    def _record_history(self):
        """Record current state metrics to history."""
        if self.history is None:
            return  # Skip recording in cloned environments
        self.history["step"].append(self.step_count)
        self.history["macrophage_health"].append(self.macrophage.health)
        self.history["bacteria_count"].append(len(self.bacteria))
        self.history["nutrients_count"].append(len(self.nutrients))

    def get_grid(self):
        """Return a 2D numpy array representing the grid state with codes for each cell."""
        grid = np.zeros((self.size, self.size), dtype=int)
        # Mark nutrients
        for (nx, ny) in self.nutrients:
            grid[ny, nx] = 3  # nutrient
        # Mark bacteria
        for b in self.bacteria:
            x, y = b.position
            grid[y, x] = 2  # bacteria
        # Mark macrophage (could override nutrient if same cell, but Mac removes nutrient when stepping on it)
        mx, my = self.macrophage.position
        grid[my, mx] = 1  # macrophage
        return grid

    def step(self, macrophage_action=None):
        """
        Advance the simulation by one time step.
        If macrophage_action is provided, use it instead of computing via the agent (used for MCTS simulation).
        """
        if self.done:
            return  # if simulation is already finished, do nothing
        # Macrophage decision
        action = macrophage_action
        if action is None:
            # Use MCTS or heuristic to choose action
            from agents.mcts import MCTSAgent
            agent = MCTSAgent()
            action = agent.choose_action(self)
        # Execute Macrophage action
        self._execute_macrophage_action(action)
        # Check if Macrophage's action eliminated all bacteria (could happen if toxin kills all)
        if len(self.bacteria) == 0:
            self.done = True
            self.winner = "Macrophage"
            self.win_reason = "eliminated all bacteria"
        # Bacteria actions (only if simulation not ended by Macrophage action)
        if not self.done:
            self._execute_bacteria_actions()
        # End of step: update history, step counter
        self.step_count += 1
        self._record_history()
        # Record Macrophage position frequency
        if self.macrophage_position_frequency is not None:
            mx, my = self.macrophage.position
            self.macrophage_position_frequency[my, mx] += 1
        # Check win conditions after bacteria actions
        if not self.done:
            # Check Macrophage death
            if self.macrophage.health <= 0:
                self.done = True
                self.winner = "Bacteria"
                self.win_reason = "macrophage died"
            # Check time horizon reached with bacteria surviving
            elif self.step_count >= config.TIME_HORIZON and len(self.bacteria) > 0:
                self.done = True
                self.winner = "Bacteria"
                self.win_reason = "time horizon reached"
            # Check bacteria population explosion
            elif len(self.bacteria) >= config.BACTERIA_MAX_COUNT:
                self.done = True
                self.winner = "Bacteria"
                self.win_reason = "bacteria overran (population >= %d)" % config.BACTERIA_MAX_COUNT
        return action  # return the action taken by Macrophage for logging if needed

    def _execute_macrophage_action(self, action):
        """
        Apply the Macrophage's chosen action to the environment.
        action format: either ("move", (dx, dy)) or ("toxin", None).
        """
        atype, param = action
        if atype == "move":
            dx, dy = param
            # Compute new position
            mx, my = self.macrophage.position
            new_x = mx + dx
            new_y = my + dy
            # Bound check
            if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
                # If move is invalid (shouldn't happen if agent is well-behaved), ignore it
                new_x, new_y = mx, my
            # Update position
            self.macrophage.position = (new_x, new_y)
            # If Macrophage moves onto a bacteria, that bacteria is eliminated
            self.bacteria = [b for b in self.bacteria if b.position != (new_x, new_y)]
            # If Macrophage moves onto a nutrient, remove the nutrient (no effect on Macrophage health)
            if (new_x, new_y) in self.nutrients:
                self.nutrients.remove((new_x, new_y))
        elif atype == "toxin":
            # Check toxin cooldown
            if config.MACROPHAGE_EXHAUSTION_ENABLED:
                if self.step_count - self.macrophage.last_toxin_step < config.MACROPHAGE_TOXIN_COOLDOWN:
                    # Toxin on cooldown, cannot use
                    return
            
            # Macrophage uses toxin burst at current position
            mx, my = self.macrophage.position
            # Affect all bacteria within toxin radius
            killed_count = 0
            
            bacteria_to_keep = []
            for b in self.bacteria:
                bx, by = b.position
                # Manhattan distance
                if abs(bx - mx) <= config.MACROPHAGE_TOXIN_RADIUS and abs(by - my) <= config.MACROPHAGE_TOXIN_RADIUS:
                    # Calculate damage (reduced by biofilm)
                    damage = config.MACROPHAGE_TOXIN_DAMAGE
                    if config.QUORUM_SENSING_ENABLED and b.in_biofilm:
                        damage = max(1, damage - config.BIOFILM_DEFENSE_BONUS)
                    
                    b.health -= damage
                    
                    # Mark toxin effect on that cell
                    if self.toxin_effect_frequency is not None:
                        self.toxin_effect_frequency[by, bx] += 1
                    
                    if b.health <= 0:
                        killed_count += 1
                    else:
                        bacteria_to_keep.append(b)
                else:
                    bacteria_to_keep.append(b)
            
            # Update bacteria list
            self.bacteria = bacteria_to_keep
            
            # Update macrophage state after using toxin
            if config.MACROPHAGE_EXHAUSTION_ENABLED:
                self.macrophage.last_toxin_step = self.step_count
                self.macrophage.exhaustion += killed_count * config.MACROPHAGE_EXHAUSTION_PER_KILL
        else:
            # Unknown action type, do nothing
            pass

    def _execute_bacteria_actions(self):
        """Process all bacteria actions (movement, replication, attacks) for this time step."""
        # Age all bacteria and check for quorum sensing/biofilm formation
        self._update_bacteria_state()
        
        # Process virulence factors (passive toxin damage to macrophage)
        if config.VIRULENCE_ENABLED:
            self._apply_virulence_damage()
        
        # First, bacteria attacks if adjacent to Macrophage
        macrophage_adjacent = []
        mx, my = self.macrophage.position
        for b in self.bacteria:
            bx, by = b.position
            if abs(bx - mx) <= 1 and abs(by - my) <= 1:
                # Bacteria is adjacent to Macrophage
                macrophage_adjacent.append(b)
                
                # Only mature bacteria can attack effectively
                if b.age < config.BACTERIA_MATURITY_AGE:
                    continue  # Immature bacteria cannot attack
                
                # Attack with probability
                if random.random() < config.BACTERIA_ATTACK_PROB:
                    damage = config.BACTERIA_ATTACK_DAMAGE
                    
                    # Biofilm bonus damage
                    if config.QUORUM_SENSING_ENABLED and b.in_biofilm:
                        damage += config.BIOFILM_ATTACK_BONUS
                    
                    # Exhausted macrophage takes more damage
                    if config.MACROPHAGE_EXHAUSTION_ENABLED:
                        if self.macrophage.exhaustion > config.MACROPHAGE_EXHAUSTION_DAMAGE_THRESHOLD:
                            damage += 5
                    
                    self.macrophage.health -= damage
        
        # Macrophage exhaustion recovery when not surrounded
        if config.MACROPHAGE_EXHAUSTION_ENABLED:
            if len(macrophage_adjacent) == 0:
                self.macrophage.exhaustion = max(0, self.macrophage.exhaustion - config.MACROPHAGE_EXHAUSTION_RECOVERY)
        
        # If Macrophage died from attacks, we can early exit (though we will catch in step end)
        # Next, handle movement and replication for bacteria that were not adjacent (they already "acted" by attacking)
        new_bacteria = []
        for b in self.bacteria:
            if b in macrophage_adjacent:
                continue  # skip those that attacked (no movement this turn)
            # Determine replication chance (boost if had nutrient last turn)
            # If the bacteria consumed a nutrient in the previous move, it might have a boost
            prob = config.BACTERIA_REPLICATION_PROB
            if b.nutrient_boost:
                prob += config.BACTERIA_REPLICATION_PROB_BOOST
            # Reset nutrient boost flag for next turn (it only applies once)
            b.nutrient_boost = False
            did_replicate = False
            if random.random() < prob:
                # Attempt replication
                # Find an adjacent empty cell for new bacteria
                dirs = [(-1, -1), (-1, 0), (-1, 1),
                        (0, -1),           (0, 1),
                        (1, -1),  (1, 0),  (1, 1)]
                random.shuffle(dirs)
                for dx, dy in dirs:
                    nx = b.position[0] + dx
                    ny = b.position[1] + dy
                    # Check valid and empty (no Macrophage and no other bacteria)
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        # ensure no bacteria currently at (nx, ny)
                        occupied = False
                        for other in self.bacteria:
                            if other.position == (nx, ny):
                                occupied = True
                                break
                        if occupied:
                            continue
                        # also ensure Macrophage not at that position
                        if (nx, ny) == self.macrophage.position:
                            continue
                        # also avoid nutrient check for occupancy? Actually, bacteria can spawn on nutrient and consume it immediately maybe
                        # We'll allow spawning on nutrient (which would mean immediate consumption next section).
                        # Create new bacteria
                        nb = Bacteria(position=(nx, ny), health=config.BACTERIA_INITIAL_HEALTH_WEAK)
                        new_bacteria.append(nb)
                        did_replicate = True
                        break
            if not did_replicate:
                # Move bacteria according to behavior mode
                dx, dy = self._choose_bacteria_move_direction(b)
                if dx is None or dy is None:
                    # No move (e.g., no available or chosen to stay)
                    pass
                else:
                    new_x = b.position[0] + dx
                    new_y = b.position[1] + dy
                    # Bound and occupancy check
                    if 0 <= new_x < self.size and 0 <= new_y < self.size:
                        # ensure target is not occupied by another bacteria or Macrophage
                        occupied = False
                        for other in self.bacteria:
                            if other is b:
                                continue
                            if other.position == (new_x, new_y):
                                occupied = True
                                break
                        if (new_x, new_y) == self.macrophage.position:
                            occupied = True
                        if not occupied:
                            b.position = (new_x, new_y)
                            # If moved onto a nutrient, consume it
                            if (new_x, new_y) in self.nutrients:
                                self.nutrients.remove((new_x, new_y))
                                # Heal the bacteria
                                b.health = min(config.BACTERIA_MAX_HEALTH, b.health + config.NUTRIENT_HEAL_AMOUNT)
                                # Set nutrient boost flag to increase replication chance next turn
                                b.nutrient_boost = True
        # Add any new bacteria produced by replication this turn
        self.bacteria.extend(new_bacteria)
        # Health regeneration for under-strength bacteria
        for b in self.bacteria:
            if b.health < config.BACTERIA_MAX_HEALTH:
                b.health = min(config.BACTERIA_MAX_HEALTH, b.health + config.BACTERIA_HEALTH_REGEN)
    
    def _update_bacteria_state(self):
        """Update bacterial age, biofilm status, and virulence activation."""
        # Age all bacteria
        for b in self.bacteria:
            b.age += 1
        
        # Check for quorum sensing and biofilm formation
        if config.QUORUM_SENSING_ENABLED:
            for b in self.bacteria:
                # Count nearby bacteria
                nearby_count = 0
                for other in self.bacteria:
                    if other is b:
                        continue
                    dist = abs(other.position[0] - b.position[0]) + abs(other.position[1] - b.position[1])
                    if dist <= config.QUORUM_RADIUS:
                        nearby_count += 1
                
                # Activate biofilm if quorum threshold met
                b.in_biofilm = (nearby_count >= config.QUORUM_THRESHOLD)
        
        # Activate virulence factors for mature bacteria
        if config.VIRULENCE_ENABLED:
            for b in self.bacteria:
                if b.age >= config.BACTERIA_MATURITY_AGE:
                    # Mature bacteria can activate virulence
                    if not b.virulence_active and random.random() < config.VIRULENCE_ACTIVATION_PROB:
                        b.virulence_active = True
    
    def _apply_virulence_damage(self):
        """Apply passive toxin damage from virulent bacteria to macrophage."""
        mx, my = self.macrophage.position
        virulent_nearby = 0
        
        for b in self.bacteria:
            if not b.virulence_active:
                continue
            
            bx, by = b.position
            dist = abs(bx - mx) + abs(by - my)
            
            if dist <= config.VIRULENCE_RADIUS:
                virulent_nearby += 1
        
        # Apply cumulative toxin damage
        if virulent_nearby > 0:
            damage = virulent_nearby * config.VIRULENCE_DAMAGE_PER_STEP
            self.macrophage.health -= damage

    def _choose_bacteria_move_direction(self, b):
        """
        Determine movement direction (dx, dy) for a bacteria based on the current behavior mode.
        Returns a tuple (dx, dy) or (None, None) for no movement.
        """
        # Use adaptive AI if enabled
        if config.BACTERIA_ADAPTIVE_ENABLED:
            from agents.bacteria_adaptive import choose_bacteria_move
            return choose_bacteria_move(b, self)
        
        mode = config.BACTERIA_MODE.lower()
        bx, by = b.position
        # Helper for moving toward a target point
        def toward(target_x, target_y):
            dx = 0
            dy = 0
            if bx < target_x: dx = 1
            elif bx > target_x: dx = -1
            if by < target_y: dy = 1
            elif by > target_y: dy = -1
            return dx, dy
        if mode == "cluster":
            # move toward the center of mass of all bacteria
            if len(self.bacteria) > 1:
                avg_x = sum(bb.position[0] for bb in self.bacteria if bb is not b) / (len(self.bacteria) - 1)
                avg_y = sum(bb.position[1] for bb in self.bacteria if bb is not b) / (len(self.bacteria) - 1)
                target_x = int(round(avg_x))
                target_y = int(round(avg_y))
                return toward(target_x, target_y)
            else:
                return (0, 0)  # alone, no movement
        elif mode == "scatter":
            # move in a random direction (including staying put)
            dirs = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
            dx, dy = random.choice(dirs)
            return dx, dy
        elif mode == "defend":
            # If macrophage is within a certain range, move toward it (aggressively engage)
            mx, my = self.macrophage.position
            dist = abs(bx - mx) + abs(by - my)
            if dist <= 6:  # if macrophage is relatively near, move toward it
                return toward(mx, my)
            else:
                # otherwise, cluster with others (defensive grouping)
                if len(self.bacteria) > 1:
                    avg_x = sum(bb.position[0] for bb in self.bacteria if bb is not b) / (len(self.bacteria) - 1)
                    avg_y = sum(bb.position[1] for bb in self.bacteria if bb is not b) / (len(self.bacteria) - 1)
                    target_x = int(round(avg_x))
                    target_y = int(round(avg_y))
                    return toward(target_x, target_y)
                else:
                    return (0, 0)
        elif mode == "replicate":
            # Prioritize moving toward nutrients if any available
            if self.nutrients:
                # find nearest nutrient
                nearest = None
                nearest_dist = float('inf')
                for (nx, ny) in self.nutrients:
                    d = abs(bx - nx) + abs(by - ny)
                    if d < nearest_dist:
                        nearest_dist = d
                        nearest = (nx, ny)
                if nearest:
                    nx, ny = nearest
                    return toward(nx, ny)
            # If no nutrients or not moving towards them, scatter to explore
            dirs = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
            dx, dy = random.choice(dirs)
            return dx, dy
        else:
            # default behavior (if mode unrecognized): random move
            dirs = [(0,0), (1,0), (-1,0), (0,1), (0,-1)]
            dx, dy = random.choice(dirs)
            return dx, dy

    def clone(self):
        """Create a deep copy of the environment (for simulation in MCTS)."""
        env_clone = Environment.__new__(Environment)  # create uninitialized instance
        # Copy simple attributes
        env_clone.size = self.size
        env_clone.step_count = self.step_count
        env_clone.done = self.done
        env_clone.winner = self.winner
        env_clone.win_reason = self.win_reason
        # Deep copy Macrophage and Bacteria with all new fields
        env_clone.macrophage = Macrophage(
            position=self.macrophage.position, 
            health=self.macrophage.health,
            exhaustion=self.macrophage.exhaustion,
            last_toxin_step=self.macrophage.last_toxin_step
        )
        env_clone.bacteria = [
            Bacteria(
                position=b.position, 
                health=b.health, 
                nutrient_boost=b.nutrient_boost,
                age=b.age,
                in_biofilm=b.in_biofilm,
                virulence_active=b.virulence_active
            ) for b in self.bacteria
        ]
        # Copy nutrients set
        env_clone.nutrients = set(self.nutrients)
        # Note: history and frequency maps not needed for simulation clone (we skip logging in rollouts)
        env_clone.history = None
        env_clone.macrophage_position_frequency = None
        env_clone.toxin_effect_frequency = None
        return env_clone

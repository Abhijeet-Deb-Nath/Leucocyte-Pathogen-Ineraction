"""
Adaptive bacteria agent with dynamic behavior based on environmental conditions.
Replaces hardcoded behavior modes with context-aware decision making.
"""
import random
import config


class AdaptiveBacteriaAgent:
    """
    Intelligent bacteria decision-making based on local context.
    Each bacterium evaluates its situation and chooses strategy dynamically.
    """
    
    def __init__(self, stochasticity=0.2):
        """
        Args:
            stochasticity: Probability of random action (0.0 = deterministic, 1.0 = pure random)
        """
        self.stochasticity = stochasticity
    
    def choose_action(self, bacterium, env):
        """
        Choose movement direction for a bacterium based on adaptive strategy.
        
        Args:
            bacterium: The Bacteria object making the decision
            env: The Environment object
            
        Returns:
            (dx, dy): Movement direction tuple
        """
        # Add stochastic noise - sometimes make random moves
        if random.random() < self.stochasticity:
            return self._random_move()
        
        # Analyze situation and choose strategy
        strategy = self._assess_situation(bacterium, env)
        
        # Execute chosen strategy
        if strategy == "flee":
            return self._flee_from_macrophage(bacterium, env)
        elif strategy == "seek_nutrient":
            return self._seek_nearest_nutrient(bacterium, env)
        elif strategy == "attack":
            return self._pursue_macrophage(bacterium, env)
        elif strategy == "cluster":
            return self._move_to_cluster(bacterium, env)
        elif strategy == "explore":
            return self._explore_cautiously(bacterium, env)
        else:
            return self._random_move()
    
    def _assess_situation(self, bacterium, env):
        """
        Analyze the bacterium's context to choose optimal strategy.
        
        Decision tree:
        1. Low health + nutrient nearby → seek_nutrient
        2. Macrophage very close + no biofilm → flee
        3. Macrophage close + in biofilm + mature → attack
        4. Quorum threshold met → cluster (form biofilm)
        5. Default → explore
        """
        bx, by = bacterium.position
        mx, my = env.macrophage.position
        
        # Calculate macrophage distance
        macrophage_dist = abs(bx - mx) + abs(by - my)
        
        # Check if low health
        low_health = bacterium.health < config.BACTERIA_MAX_HEALTH * 0.5
        
        # Check for nearby nutrients
        nearby_nutrient = self._has_nearby_nutrient(bacterium, env, radius=4)
        
        # Check maturity
        is_mature = bacterium.age >= config.BACTERIA_MATURITY_AGE
        
        # Check biofilm status
        in_biofilm = bacterium.in_biofilm
        
        # Decision logic
        if low_health and nearby_nutrient:
            return "seek_nutrient"
        
        if macrophage_dist <= 3 and not in_biofilm:
            # Macrophage is dangerously close and we're vulnerable
            return "flee"
        
        if macrophage_dist <= 5 and in_biofilm and is_mature:
            # We're in a biofilm and can fight - be aggressive
            return "attack"
        
        if self._should_cluster(bacterium, env):
            # Quorum sensing triggers clustering
            return "cluster"
        
        # Default: explore to find nutrients and avoid being isolated
        return "explore"
    
    def _has_nearby_nutrient(self, bacterium, env, radius=4):
        """Check if there's a nutrient within radius."""
        bx, by = bacterium.position
        for nx, ny in env.nutrients:
            dist = abs(bx - nx) + abs(by - ny)
            if dist <= radius:
                return True
        return False
    
    def _should_cluster(self, bacterium, env):
        """Check if quorum sensing suggests clustering."""
        if not config.QUORUM_SENSING_ENABLED:
            return False
        
        bx, by = bacterium.position
        nearby_count = 0
        
        for other in env.bacteria:
            if other is bacterium:
                continue
            ox, oy = other.position
            dist = abs(bx - ox) + abs(by - oy)
            if dist <= config.QUORUM_RADIUS:
                nearby_count += 1
        
        # If close to quorum threshold, cluster to form biofilm
        return nearby_count >= config.QUORUM_THRESHOLD - 1
    
    def _flee_from_macrophage(self, bacterium, env):
        """Move away from macrophage."""
        bx, by = bacterium.position
        mx, my = env.macrophage.position
        
        dx = 0
        dy = 0
        
        # Move opposite direction from macrophage
        if bx < mx:
            dx = -1  # Move left
        elif bx > mx:
            dx = 1   # Move right
        
        if by < my:
            dy = -1  # Move down
        elif by > my:
            dy = 1   # Move up
        
        return (dx, dy)
    
    def _seek_nearest_nutrient(self, bacterium, env):
        """Move toward closest nutrient."""
        if not env.nutrients:
            return self._explore_cautiously(bacterium, env)
        
        bx, by = bacterium.position
        
        # Find nearest nutrient
        nearest_nutrient = min(
            env.nutrients,
            key=lambda pos: abs(pos[0] - bx) + abs(pos[1] - by)
        )
        
        nx, ny = nearest_nutrient
        return self._move_toward(bx, by, nx, ny)
    
    def _pursue_macrophage(self, bacterium, env):
        """Move toward macrophage (aggressive)."""
        bx, by = bacterium.position
        mx, my = env.macrophage.position
        return self._move_toward(bx, by, mx, my)
    
    def _move_to_cluster(self, bacterium, env):
        """Move toward center of mass of nearby bacteria."""
        if len(env.bacteria) <= 1:
            return (0, 0)
        
        bx, by = bacterium.position
        
        # Calculate center of mass of other bacteria
        other_bacteria = [b for b in env.bacteria if b is not bacterium]
        avg_x = sum(b.position[0] for b in other_bacteria) / len(other_bacteria)
        avg_y = sum(b.position[1] for b in other_bacteria) / len(other_bacteria)
        
        target_x = int(round(avg_x))
        target_y = int(round(avg_y))
        
        return self._move_toward(bx, by, target_x, target_y)
    
    def _explore_cautiously(self, bacterium, env):
        """
        Random exploration but avoid moving toward macrophage if too close.
        """
        bx, by = bacterium.position
        mx, my = env.macrophage.position
        macrophage_dist = abs(bx - mx) + abs(by - my)
        
        # If macrophage is far, just move randomly
        if macrophage_dist > 6:
            return self._random_move()
        
        # If macrophage is medium distance, prefer moves that don't get closer
        possible_moves = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        
        # Filter moves that don't decrease distance to macrophage
        safe_moves = []
        for dx, dy in possible_moves:
            new_x, new_y = bx + dx, by + dy
            new_dist = abs(new_x - mx) + abs(new_y - my)
            if new_dist >= macrophage_dist:  # Not getting closer
                safe_moves.append((dx, dy))
        
        if safe_moves:
            return random.choice(safe_moves)
        else:
            # All moves get closer, so flee instead
            return self._flee_from_macrophage(bacterium, env)
    
    def _move_toward(self, from_x, from_y, to_x, to_y):
        """Calculate move direction toward target."""
        dx = 0
        dy = 0
        
        if from_x < to_x:
            dx = 1
        elif from_x > to_x:
            dx = -1
        
        if from_y < to_y:
            dy = 1
        elif from_y > to_y:
            dy = -1
        
        return (dx, dy)
    
    def _random_move(self):
        """Random movement including staying in place."""
        moves = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        return random.choice(moves)


# Global singleton for easy access
adaptive_bacteria_agent = AdaptiveBacteriaAgent(stochasticity=0.15)


def choose_bacteria_move(bacterium, env):
    """
    Wrapper function to integrate with existing environment code.
    
    Usage in environment.py:
        from agents.bacteria_adaptive import choose_bacteria_move
        dx, dy = choose_bacteria_move(bacterium, env)
    """
    return adaptive_bacteria_agent.choose_action(bacterium, env)

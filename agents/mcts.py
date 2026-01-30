import random
import time
import config

class MCTSAgent:
    """
    Monte Carlo Tree Search based agent for the Macrophage.
    Performs rollouts to evaluate possible actions and picks the best action.
    """
    def __init__(self, rollout_budget=None, rollout_depth=None):
        self.rollout_budget = rollout_budget if rollout_budget is not None else config.ROLLOUT_BUDGET
        self.rollout_depth = rollout_depth if rollout_depth is not None else config.MCTS_SIM_DEPTH
        # If rollout_budget is zero or negative, we can skip MCTS and use heuristic directly.
        from agents.heuristic import HeuristicAgent
        self.fallback_agent = HeuristicAgent()

    def choose_action(self, env):
        # If no rollouts allowed, use fallback heuristic
        if self.rollout_budget <= 0:
            return self.fallback_agent.choose_action(env)
        # Define possible actions (Macrophage can move or use toxin)
        # Moves: up, down, left, right
        actions = [
            ("move", (-1, 0)),  # left
            ("move", (1, 0)),   # right
            ("move", (0, -1)),  # down
            ("move", (0, 1)),   # up
            ("toxin", None)
        ]
        # Filter out moves that would go out of bounds
        mx, my = env.macrophage.position
        filtered_actions = []
        for action in actions:
            atype, param = action
            if atype == "move":
                dx, dy = param
                nx, ny = mx + dx, my + dy
                if nx < 0 or nx >= env.size or ny < 0 or ny >= env.size:
                    continue  # skip invalid move
            filtered_actions.append(action)
        if not filtered_actions:
            filtered_actions = [("move", (0, 0))]  # no valid moves (should not happen), use no-op
        # Run rollouts to evaluate each action
        best_action = None
        best_score = -float('inf')
        start_time = time.time()
        for action in filtered_actions:
            total_score = 0.0
            # Number of rollouts for this action
            # We can divide budget evenly among actions (at least one each)
            num_rollouts = max(1, self.rollout_budget // len(filtered_actions))
            for r in range(num_rollouts):
                # Check time budget (if too slow, fallback)
                if time.time() - start_time > 2.0:  # 2 seconds cutoff for MCTS
                    # Time budget exceeded, use fallback
                    return self.fallback_agent.choose_action(env)
                # Clone the environment to simulate
                env_clone = env.clone()
                # Apply the candidate action for Macrophage on the cloned env
                env_clone.step(macrophage_action=action)
                # Simulate further steps up to rollout_depth
                steps = 1
                while steps < self.rollout_depth and not env_clone.done:
                    # Macrophage random action (random policy for simulation)
                    rand_action = random.choice(filtered_actions)
                    env_clone.step(macrophage_action=rand_action)
                    steps += 1
                # Evaluate the outcome of this rollout
                score = self._evaluate_state(env_clone)
                total_score += score
            avg_score = total_score / num_rollouts
            if avg_score > best_score:
                best_score = avg_score
                best_action = action
        # If no action was selected (should not happen since we had fallback above), use heuristic
        if best_action is None:
            return self.fallback_agent.choose_action(env)
        return best_action

    def _evaluate_state(self, env):
        """Compute a heuristic score for a given simulation state (env)."""
        if env.winner == "Macrophage":
            # Macrophage win is a very good outcome
            return 1000.0
        if env.winner == "Bacteria":
            # Macrophage died or time out (bacteria win) is a very bad outcome
            return -1000.0
        # If not terminal, use number of bacteria and macrophage health as metrics
        score = 0.0
        score += env.macrophage.health  # higher health is better
        score -= len(env.bacteria) * 10.0  # fewer bacteria is better
        return score

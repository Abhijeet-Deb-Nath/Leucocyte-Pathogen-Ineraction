import random
import time

import config
from agents.heuristic import HeuristicAgent


class MCTSAgent:
    """Lightweight rollout planner over Pathway 1 host action space."""

    def __init__(self, rollout_budget=None, rollout_depth=None):
        self.rollout_budget = rollout_budget if rollout_budget is not None else config.ROLLOUT_BUDGET
        self.rollout_depth = rollout_depth if rollout_depth is not None else config.MCTS_SIM_DEPTH
        self.fallback_agent = HeuristicAgent()

    def choose_action(self, env):
        actions = env.get_macrophage_actions()
        if not actions:
            return ("move", (0, 0))

        if self.rollout_budget <= 0:
            return self.fallback_agent.choose_action(env)

        best_action = None
        best_score = -float("inf")
        start_time = time.time()

        rollouts_per_action = max(1, self.rollout_budget // len(actions))

        for action in actions:
            cumulative = 0.0
            for _ in range(rollouts_per_action):
                if time.time() - start_time > 1.5:
                    return self.fallback_agent.choose_action(env)

                sim = env.clone()
                sim.step(macrophage_action=action)

                depth = 1
                while depth < self.rollout_depth and not sim.done:
                    random_action = random.choice(sim.get_macrophage_actions())
                    sim.step(macrophage_action=random_action)
                    depth += 1

                cumulative += self._evaluate(sim)

            avg = cumulative / rollouts_per_action
            if avg > best_score:
                best_score = avg
                best_action = action

        return best_action if best_action is not None else self.fallback_agent.choose_action(env)

    def _evaluate(self, env):
        if env.winner == "Host":
            return 1000.0 + env.compute_host_utility()
        if env.winner == "Bacteria":
            return -1000.0 + env.compute_host_utility()

        # Mid-rollout estimate combining current burden and utility trajectory.
        return (
            env.compute_host_utility()
            - 2.5 * len(env.bacteria)
            - 0.75 * len(env.neutrophils)
            + 0.2 * env.macrophage.health
        )

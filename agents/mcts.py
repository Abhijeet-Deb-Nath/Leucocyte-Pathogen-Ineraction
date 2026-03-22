import config
from agents.heuristic import HeuristicAgent


class MCTSAgent:
    """Lightweight rollout planner over Pathway 1 host action space."""

    def __init__(self, rollout_budget=None, rollout_depth=None):
        self.rollout_budget = rollout_budget if rollout_budget is not None else config.ROLLOUT_BUDGET
        self.rollout_depth = rollout_depth if rollout_depth is not None else config.MCTS_SIM_DEPTH
        self.fallback_agent = HeuristicAgent()
        self.sense_radius = config.MACROPHAGE_SENSE_RADIUS
        self.previous_position = None

    def reset(self, env):
        self.fallback_agent.reset(env)
        self.previous_position = None

    def choose_action(self, env):
        actions = env.get_macrophage_actions()
        if not actions:
            return ("move", (0, 0))

        visible_bacteria = env.get_local_bacteria(env.macrophage.position, self.sense_radius)
        local_peak = self._local_chemokine_peak(env)
        if not visible_bacteria and local_peak < config.MACROPHAGE_HOLD_LOCAL_CHEMOKINE_THRESHOLD:
            action = self.fallback_agent.choose_action(env)
            if action[0] == "move":
                self.previous_position = env.macrophage.position
            else:
                self.previous_position = None
            return action

        if self.rollout_budget <= 0:
            return self.fallback_agent.choose_action(env)

        best_action = None
        best_score = -float("inf")
        rollouts_per_action = max(1, self.rollout_budget // len(actions))

        for action in actions:
            cumulative = 0.0
            for _ in range(rollouts_per_action):
                sim = self._local_rollout_env(env)
                rollout_agent = HeuristicAgent()
                rollout_agent.reset(sim)
                sim.step(macrophage_action=action)

                depth = 1
                while depth < self.rollout_depth and not sim.done:
                    rollout_action = rollout_agent.choose_action(sim)
                    sim.step(macrophage_action=rollout_action)
                    depth += 1

                cumulative += self._evaluate(sim) + self._action_prior(env, action)

            avg = cumulative / rollouts_per_action
            if avg > best_score:
                best_score = avg
                best_action = action

        if best_action is None:
            best_action = self.fallback_agent.choose_action(env)
        if best_action[0] == "move":
            self.previous_position = env.macrophage.position
        else:
            self.previous_position = None
        return best_action

    def _evaluate(self, env):
        if env.winner == "Host":
            return 1000.0 + env.compute_host_utility()
        if env.winner == "Bacteria":
            return -1000.0 + env.compute_host_utility()

        visible_bacteria = env.get_local_bacteria(env.macrophage.position, self.sense_radius)
        visible_neutrophils = env.get_local_neutrophils(env.macrophage.position, self.sense_radius)
        adjacent_bacteria = len(env._adjacent_bacteria(env.macrophage.position))
        local_peak = self._local_chemokine_peak(env)

        # Mid-rollout estimate using only locally observable tactical pressure.
        return (
            env.compute_host_utility()
            - 3.0 * len(visible_bacteria)
            - 0.5 * len(visible_neutrophils)
            + 1.5 * adjacent_bacteria
            + 0.25 * local_peak
            + 0.2 * env.macrophage.health
        )

    def _local_rollout_env(self, env):
        # Keep the generative dynamics intact so local rollouts can discover
        # infection by moving into it, while the rollout policy itself remains
        # local-information driven.
        return env.clone()

    def _action_prior(self, env, action):
        current_pos = env.macrophage.position
        visible_bacteria = env.get_local_bacteria(current_pos, self.sense_radius)
        visible_neutrophils = env.get_local_neutrophils(current_pos, self.sense_radius)
        local_peak = self._local_chemokine_peak(env)
        nearest_visible_distance = None
        if visible_bacteria:
            nearest_visible_distance = min(
                abs(b.position[0] - current_pos[0]) + abs(b.position[1] - current_pos[1])
                for b in visible_bacteria
            )

        if action == ("attack", None):
            return 8.0 if env._adjacent_bacteria(current_pos) else -4.0

        if action[0] in {"signal", "signal_low", "signal_medium", "signal_high"}:
            pressure = len(visible_bacteria) - len(visible_neutrophils)
            if env._adjacent_bacteria(current_pos):
                return -2.0
            if (
                visible_bacteria
                and nearest_visible_distance is not None
                and nearest_visible_distance <= config.HEURISTIC_SIGNAL_PROXIMITY_RADIUS
                and pressure >= config.HEURISTIC_SIGNAL_PRESSURE_MEDIUM
            ):
                return 0.8
            if (
                local_peak >= config.MACROPHAGE_SIGNAL_LOCAL_CHEMOKINE_THRESHOLD
                and pressure >= config.HEURISTIC_SIGNAL_PRESSURE_HIGH
            ):
                return 1.2
            return -3.0

        if action[0] != "move":
            return 0.0

        dx, dy = action[1]
        next_pos = (current_pos[0] + dx, current_pos[1] + dy)
        score = 0.0
        if (dx, dy) == (0, 0):
            score -= 1.0 if visible_bacteria or local_peak > 0.05 else 0.3
        if self.previous_position is not None and next_pos == self.previous_position:
            score -= 1.5
        if next_pos in env.doorway_tiles:
            score += 0.4
        score += min(float(env.chemokine[next_pos[1], next_pos[0]]), config.MACROPHAGE_CHEM_SENSE_SCALE) * 0.75

        if visible_bacteria:
            nearest_before = min(
                abs(b.position[0] - current_pos[0]) + abs(b.position[1] - current_pos[1])
                for b in visible_bacteria
            )
            nearest_after = min(
                abs(b.position[0] - next_pos[0]) + abs(b.position[1] - next_pos[1])
                for b in visible_bacteria
            )
            score += 1.2 * (nearest_before - nearest_after)
            if nearest_after <= 1:
                score += 0.8
        return score

    def _local_chemokine_peak(self, env):
        mx, my = env.macrophage.position
        peak = 0.0
        for y in range(max(0, my - self.sense_radius), min(env.height, my + self.sense_radius + 1)):
            for x in range(max(0, mx - self.sense_radius), min(env.width, mx + self.sense_radius + 1)):
                if abs(x - mx) + abs(y - my) > self.sense_radius:
                    continue
                peak = max(peak, float(env.chemokine[y, x]))
        return peak

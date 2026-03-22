from collections import deque

import config


class HeuristicAgent:
    """Pathway 1 host policy with explicit contain-vs-recruit trade-off."""

    def __init__(self, sense_radius=None, use_memory=True, stochastic_patrol=True):
        self.sense_radius = (
            config.MACROPHAGE_SENSE_RADIUS if sense_radius is None else int(sense_radius)
        )
        self.use_memory = bool(use_memory)
        self.stochastic_patrol = bool(stochastic_patrol)
        self.recent_positions = deque(maxlen=6)
        self.visited_positions = set()

    def reset(self, env):
        self.recent_positions = deque([env.macrophage.position], maxlen=6)
        self.visited_positions = {env.macrophage.position}

    def choose_action(self, env):
        if not self.recent_positions:
            self.reset(env)
        if self.use_memory:
            self.visited_positions.add(env.macrophage.position)
        actions = env.get_macrophage_actions()
        action_set = set(actions)

        mx, my = env.macrophage.position
        nearby_bacteria = env.get_local_bacteria((mx, my), self.sense_radius)
        nearby_neutrophils = env.get_local_neutrophils((mx, my), self.sense_radius)
        adjacent_bacteria = [
            b for b in nearby_bacteria if abs(b.position[0] - mx) + abs(b.position[1] - my) <= 1
        ]

        if adjacent_bacteria and ("attack", None) in action_set:
            return ("attack", None)

        if nearby_bacteria:
            target = min(
                nearby_bacteria,
                key=lambda b: abs(b.position[0] - mx) + abs(b.position[1] - my),
            ).position
            nearest_distance = abs(target[0] - mx) + abs(target[1] - my)
            local_peak = env.get_local_chemokine_peak(
                (mx, my),
                config.MACROPHAGE_SIGNAL_TRIGGER_RADIUS,
            )
            pressure = len(nearby_bacteria) - 0.75 * len(nearby_neutrophils)
            support_missing = len(nearby_neutrophils) == 0 and not env.recruitment_queue
            low_health = env.macrophage.health <= config.MACROPHAGE_MAX_HEALTH * 0.45

            # Favor direct engagement unless the local burden is severe or the
            # macrophage is already under clear pressure.
            if pressure >= config.HEURISTIC_SIGNAL_PRESSURE_HIGH or (
                low_health and pressure >= config.HEURISTIC_SIGNAL_PRESSURE_MEDIUM
            ):
                signal_action = self._select_signal_action(action_set, level="high")
                if signal_action is not None:
                    return signal_action
            elif (
                support_missing
                and pressure >= config.HEURISTIC_SIGNAL_PRESSURE_MEDIUM
                and nearest_distance <= config.HEURISTIC_SIGNAL_PROXIMITY_RADIUS + 1
            ):
                signal_action = self._select_signal_action(action_set, level="medium")
                if signal_action is not None:
                    return signal_action
            elif (
                support_missing
                and pressure >= config.HEURISTIC_SIGNAL_PRESSURE_LOW
                and nearest_distance <= self.sense_radius
                and local_peak >= config.MACROPHAGE_SIGNAL_LOCAL_CHEMOKINE_THRESHOLD * 0.75
            ):
                signal_action = self._select_signal_action(action_set, level="low")
                if signal_action is not None:
                    return signal_action
            return self._move_toward_target(env, target)

        # Follow only locally sensed chemokine rather than any global peak.
        chem_target = self._local_chemokine_target(env)
        if chem_target is not None:
            return self._move_toward_target(env, chem_target)

        move_actions = [a for a in actions if a[0] == "move" and a[1] != (0, 0)]
        if not move_actions:
            return ("move", (0, 0))
        return self._patrol_move(env, move_actions)

    def _local_chemokine_target(self, env):
        mx, my = env.macrophage.position
        best_target = None
        best_score = max(0.05, config.BACTERIA_ALARM_CHEMOKINE_RELEASE * 0.5)
        for dy in range(-self.sense_radius, self.sense_radius + 1):
            for dx in range(-self.sense_radius, self.sense_radius + 1):
                if abs(dx) + abs(dy) > self.sense_radius:
                    continue
                x = mx + dx
                y = my + dy
                if not (0 <= x < env.width and 0 <= y < env.height):
                    continue
                if not env._is_walkable((x, y)):
                    continue

                chem_level = float(env.chemokine[y, x])
                distance = abs(dx) + abs(dy)
                doorway_bonus = 0.15 if (x, y) in env.doorway_tiles else 0.0
                score = chem_level - 0.02 * distance + doorway_bonus
                if score > best_score:
                    best_score = score
                    best_target = (x, y)
        return best_target

    def _move_toward_target(self, env, target):
        tx, ty = target
        mx, my = env.macrophage.position

        previous_pos = (
            self.recent_positions[-2]
            if self.use_memory and len(self.recent_positions) >= 2
            else None
        )
        candidates = []
        for action in env.get_macrophage_actions():
            if action[0] != "move":
                continue
            dx, dy = action[1]
            if (dx, dy) == (0, 0) and target != (mx, my):
                continue
            nx, ny = mx + dx, my + dy
            dist = abs(nx - tx) + abs(ny - ty)
            score = 0.0
            if self.use_memory and (nx, ny) not in self.visited_positions:
                score += 0.4
            if (nx, ny) in env.doorway_tiles:
                score += 0.2
            score += min(float(env.chemokine[ny, nx]), config.MACROPHAGE_CHEM_SENSE_SCALE) * 0.35
            if previous_pos is not None and (nx, ny) == previous_pos:
                score -= 0.7
            candidates.append((dist, -score, action, (nx, ny)))

        if not candidates:
            return ("move", (0, 0))

        candidates.sort(key=lambda item: (item[0], item[1]))
        chosen_action = candidates[0][2]
        dx, dy = chosen_action[1]
        next_pos = (mx + dx, my + dy)
        self.recent_positions.append(next_pos)
        if self.use_memory:
            self.visited_positions.add(next_pos)
        return chosen_action

    def _patrol_move(self, env, move_actions):
        mx, my = env.macrophage.position
        previous_pos = (
            self.recent_positions[-2]
            if self.use_memory and len(self.recent_positions) >= 2
            else None
        )
        scored = []
        for action in move_actions:
            dx, dy = action[1]
            nx, ny = mx + dx, my + dy
            score = 0.0
            if self.use_memory and (nx, ny) not in self.visited_positions:
                score += 1.0
            if (nx, ny) in env.doorway_tiles:
                score += 0.35
            score += min(float(env.chemokine[ny, nx]), config.MACROPHAGE_CHEM_SENSE_SCALE) * 0.5
            if previous_pos is not None and (nx, ny) == previous_pos:
                score -= 0.8
            scored.append((score, action, (nx, ny)))

        best_score = max(score for score, _, _ in scored)
        best_actions = [(action, pos) for score, action, pos in scored if score >= best_score - 1e-6]
        if self.stochastic_patrol:
            chosen_action, chosen_pos = env.rng.choice(best_actions)
        else:
            chosen_action, chosen_pos = sorted(
                best_actions,
                key=lambda item: (item[0][1][0], item[0][1][1], item[1][0], item[1][1]),
            )[0]
        self.recent_positions.append(chosen_pos)
        if self.use_memory:
            self.visited_positions.add(chosen_pos)
        return chosen_action

    def _select_signal_action(self, action_set, level):
        priority = {
            "high": [
                ("signal_high", None),
                ("signal_medium", None),
                ("signal", None),
                ("signal_low", None),
            ],
            "medium": [
                ("signal_medium", None),
                ("signal", None),
                ("signal_low", None),
                ("signal_high", None),
            ],
            "low": [
                ("signal_low", None),
                ("signal_medium", None),
                ("signal", None),
                ("signal_high", None),
            ],
        }.get(level, [])
        for candidate in priority:
            if candidate in action_set:
                return candidate
        return None

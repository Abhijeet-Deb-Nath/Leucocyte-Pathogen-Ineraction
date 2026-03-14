import random

import config


class HeuristicAgent:
    """Pathway 1 host policy with explicit contain-vs-recruit trade-off."""

    def choose_action(self, env):
        actions = env.get_macrophage_actions()
        action_set = set(actions)

        mx, my = env.macrophage.position
        nearby_bacteria = env.get_local_bacteria((mx, my), config.MACROPHAGE_SENSE_RADIUS)
        adjacent_bacteria = [
            b for b in nearby_bacteria if abs(b.position[0] - mx) + abs(b.position[1] - my) <= 1
        ]

        if adjacent_bacteria and ("attack", None) in action_set:
            return ("attack", None)

        if nearby_bacteria:
            # If local burden is high and signaling is available, recruit before overcommitting.
            if len(nearby_bacteria) >= 3 and ("signal", None) in action_set:
                return ("signal", None)

            target = min(
                nearby_bacteria,
                key=lambda b: abs(b.position[0] - mx) + abs(b.position[1] - my),
            ).position
            return self._move_toward_target(env, target)

        # Patrol toward strongest chemokine region if no bacteria in local sight.
        peak = float(env.chemokine.max())
        if peak > 0.5:
            py, px = divmod(int(env.chemokine.argmax()), env.width)
            return self._move_toward_target(env, (px, py))

        move_actions = [a for a in actions if a[0] == "move"]
        if not move_actions:
            return ("move", (0, 0))
        return random.choice(move_actions)

    def _move_toward_target(self, env, target):
        tx, ty = target
        mx, my = env.macrophage.position

        candidates = []
        for action in env.get_macrophage_actions():
            if action[0] != "move":
                continue
            dx, dy = action[1]
            nx, ny = mx + dx, my + dy
            dist = abs(nx - tx) + abs(ny - ty)
            candidates.append((dist, action))

        if not candidates:
            return ("move", (0, 0))

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

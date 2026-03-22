"""Adaptive bacteria movement policy for Pathway 1 local-niche dynamics."""

import random

import config


class AdaptiveBacteriaAgent:
    def __init__(self, stochasticity=None):
        self.stochasticity = (
            config.BACTERIA_STOCHASTICITY if stochasticity is None else stochasticity
        )

    def choose_action(self, bacterium, env):
        rng = getattr(env, "rng", random)
        if rng.random() < self.stochasticity:
            return rng.choice([(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)])

        bx, by = bacterium.position
        mx, my = env.macrophage.position
        dist_to_macro = abs(bx - mx) + abs(by - my)

        local_chem = env.chemokine[by, bx]
        local_nutrient = env.nutrients[by, bx]

        # Biofilm/microcolony states avoid movement unless highly pressured.
        if bacterium.state in ("microcolony", "biofilm") and local_chem < config.NEUTROPHIL_RECRUITMENT_THRESHOLD:
            return (0, 0)

        # If immediate danger and not fortified, flee.
        if dist_to_macro <= 2 and bacterium.state == "planktonic":
            return self._move_away((bx, by), (mx, my), env)

        # If nutrients are poor, seek richer local patch.
        if local_nutrient < config.REPLICATION_REQUIRES_MIN_PATCH_RESOURCE:
            return self._seek_best_neighbor(
                bacterium.position,
                env,
                key_fn=lambda pos: env.nutrients[pos[1], pos[0]],
                maximize=True,
            )

        # If chemokine is high, drift away from strongest signal to avoid immune pressure.
        if local_chem > config.NEUTROPHIL_RECRUITMENT_THRESHOLD * 0.7:
            return self._seek_best_neighbor(
                bacterium.position,
                env,
                key_fn=lambda pos: env.chemokine[pos[1], pos[0]],
                maximize=False,
            )

        # Otherwise keep close to neighbors to support colony formation.
        return self._seek_best_neighbor(
            bacterium.position,
            env,
            key_fn=lambda pos: self._local_friend_count(pos, env),
            maximize=True,
        )

    def _move_away(self, src, threat, env):
        sx, sy = src
        tx, ty = threat
        options = env._valid_neighbors(src, include_stay=True)
        best = (0, 0)
        best_dist = -1
        for nx, ny in options:
            if any(b.position == (nx, ny) and (nx, ny) != src for b in env.bacteria):
                continue
            dist = abs(nx - tx) + abs(ny - ty)
            if dist > best_dist:
                best_dist = dist
                best = (nx - sx, ny - sy)
        return best

    def _seek_best_neighbor(self, src, env, key_fn, maximize=True):
        sx, sy = src
        options = env._valid_neighbors(src, include_stay=True)
        scored = []
        for nx, ny in options:
            if (nx, ny) != src and any(b.position == (nx, ny) for b in env.bacteria):
                continue
            scored.append((key_fn((nx, ny)), (nx - sx, ny - sy)))

        if not scored:
            return (0, 0)

        scored.sort(key=lambda item: item[0], reverse=maximize)
        return scored[0][1]

    @staticmethod
    def _local_friend_count(pos, env):
        px, py = pos
        return sum(
            1
            for b in env.bacteria
            if abs(b.position[0] - px) + abs(b.position[1] - py) <= 2
        )


adaptive_bacteria_agent = AdaptiveBacteriaAgent()


def choose_bacteria_move(bacterium, env):
    return adaptive_bacteria_agent.choose_action(bacterium, env)

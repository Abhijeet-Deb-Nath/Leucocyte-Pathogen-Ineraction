import random
from collections import deque

import numpy as np

import config
from simulator.entities import Bacteria, Macrophage, Neutrophil


class Environment:
    """Pathway 1 simulator: compartmentalized hotspot with delayed neutrophil recruitment."""

    def __init__(self, seed=None):
        self.rng = random.Random(config.SEED if seed is None else seed)
        self.size = config.GRID_SIZE
        self.width = self.size
        self.height = self.size

        self.step_count = 0
        self.done = False
        self.winner = None
        self.win_reason = None

        self.compartment_map = self._build_compartment_map()
        self.blocked_tiles = self._build_blocked_tiles()
        self.surface_patches = self._build_surface_patches()

        self.nutrients = self._initialize_nutrients()
        self.chemokine = np.zeros((self.height, self.width), dtype=float)
        self.recruitment_queue = deque()

        self.macrophage = self._spawn_macrophage()
        self.bacteria = self._spawn_bacteria()
        self.neutrophils = []

        self.killed_bacteria = 0
        self.tissue_damage = 0.0
        self.immune_usage = 0.0
        self.chemokine_events = 0
        self.current_phase = "seeding"

        self.history = {
            "step": [],
            "phase": [],
            "macrophage_health": [],
            "bacteria_count": [],
            "neutrophil_count": [],
            "tissue_damage": [],
            "host_utility": [],
            "colonized_compartments": [],
            "chemokine_peak": [],
            "nutrient_total": [],
        }

        self.macrophage_position_frequency = np.zeros((self.height, self.width), dtype=int)
        self.action_effect_frequency = np.zeros((self.height, self.width), dtype=int)

        self._record_history()

    # ---------- Initialization helpers ----------

    def _build_compartment_map(self):
        comp_map = np.full((self.height, self.width), fill_value=-1, dtype=int)
        compartment_id = 0

        for row in range(config.WORLD_ROWS):
            y0 = row * (config.COMPARTMENT_SIZE + 1)
            y1 = y0 + config.COMPARTMENT_SIZE
            for col in range(config.WORLD_COLS):
                x0 = col * (config.COMPARTMENT_SIZE + 1)
                x1 = x0 + config.COMPARTMENT_SIZE
                comp_map[y0:y1, x0:x1] = compartment_id
                compartment_id += 1

        # Carve walkable connector cells through wall rows/columns so bottleneck passages
        # are traversable, not just visually open.
        doorway_w = max(1, config.PASSAGE_BOTTLENECK_WIDTH)
        if doorway_w % 2 == 0:
            doorway_w += 1
        half = doorway_w // 2

        for row in range(config.WORLD_ROWS - 1):
            wall_y = (row + 1) * config.COMPARTMENT_SIZE + row
            cx = self.width // 2
            for dx in range(-half, half + 1):
                x = cx + dx
                if not (0 <= x < self.width):
                    continue
                if wall_y - 1 >= 0 and comp_map[wall_y - 1, x] >= 0:
                    comp_map[wall_y, x] = comp_map[wall_y - 1, x]
                elif wall_y + 1 < self.height and comp_map[wall_y + 1, x] >= 0:
                    comp_map[wall_y, x] = comp_map[wall_y + 1, x]

        for col in range(config.WORLD_COLS - 1):
            wall_x = (col + 1) * config.COMPARTMENT_SIZE + col
            cy = self.height // 2
            for dy in range(-half, half + 1):
                y = cy + dy
                if not (0 <= y < self.height):
                    continue
                if wall_x - 1 >= 0 and comp_map[y, wall_x - 1] >= 0:
                    comp_map[y, wall_x] = comp_map[y, wall_x - 1]
                elif wall_x + 1 < self.width and comp_map[y, wall_x + 1] >= 0:
                    comp_map[y, wall_x] = comp_map[y, wall_x + 1]

        return comp_map

    def _build_blocked_tiles(self):
        blocked = set()

        # Horizontal walls between compartment rows.
        for row in range(config.WORLD_ROWS - 1):
            wall_y = (row + 1) * config.COMPARTMENT_SIZE + row
            for x in range(self.width):
                blocked.add((x, wall_y))

        # Vertical walls between compartment columns.
        for col in range(config.WORLD_COLS - 1):
            wall_x = (col + 1) * config.COMPARTMENT_SIZE + col
            for y in range(self.height):
                blocked.add((wall_x, y))

        # Carve doorways through walls to create bottlenecks.
        doorway_w = max(1, config.PASSAGE_BOTTLENECK_WIDTH)
        if doorway_w % 2 == 0:
            doorway_w += 1
        half = doorway_w // 2
        for row in range(config.WORLD_ROWS - 1):
            wall_y = (row + 1) * config.COMPARTMENT_SIZE + row
            cx = self.width // 2
            for dx in range(-half, half + 1):
                blocked.discard((cx + dx, wall_y))

        for col in range(config.WORLD_COLS - 1):
            wall_x = (col + 1) * config.COMPARTMENT_SIZE + col
            cy = self.height // 2
            for dy in range(-half, half + 1):
                blocked.discard((wall_x, cy + dy))

        return blocked

    def _build_surface_patches(self):
        patches = set()
        for y in range(self.height):
            for x in range(self.width):
                if self.compartment_map[y, x] < 0:
                    continue
                if self.rng.random() < config.SURFACE_PATCH_DENSITY:
                    patches.add((x, y))
        return patches

    def _initialize_nutrients(self):
        nutrients = np.zeros((self.height, self.width), dtype=float)
        for y in range(self.height):
            for x in range(self.width):
                if self.compartment_map[y, x] >= 0:
                    nutrients[y, x] = self.rng.uniform(0.5, config.PATCH_NUTRIENT_CAPACITY)
        return nutrients

    def _spawn_macrophage(self):
        center = (self.width // 2, self.height // 2)
        if not self._is_walkable(center):
            center = self._random_walkable_cell()
        return Macrophage(position=center, health=config.MACROPHAGE_MAX_HEALTH)

    def _spawn_bacteria(self):
        bacteria = []
        # Seed bacteria in one primary hotspot compartment.
        hotspot_id = self.rng.randrange(config.N_COMPARTMENTS)
        hotspot_cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if self.compartment_map[y, x] == hotspot_id
        ]

        for _ in range(config.INITIAL_BACTERIA_COUNT):
            pos = self.rng.choice(hotspot_cells)
            while pos == self.macrophage.position or any(b.position == pos for b in bacteria):
                pos = self.rng.choice(hotspot_cells)
            bacteria.append(Bacteria(position=pos, health=config.BACTERIA_INITIAL_HEALTH))
        return bacteria

    def _random_walkable_cell(self):
        while True:
            x = self.rng.randrange(self.width)
            y = self.rng.randrange(self.height)
            if self._is_walkable((x, y)):
                return (x, y)

    # ---------- Core API ----------

    def step(self, macrophage_action=None):
        if self.done:
            return None

        action = macrophage_action
        if action is None:
            from agents.mcts import MCTSAgent

            action = MCTSAgent().choose_action(self)

        self._execute_macrophage_action(action)
        self._execute_bacteria_phase()
        self._execute_neutrophil_phase()
        self._update_fields_and_recruitment()

        self.step_count += 1
        self._update_phase_label()
        self._record_history()

        if self.macrophage_position_frequency is not None:
            mx, my = self.macrophage.position
            self.macrophage_position_frequency[my, mx] += 1

        self._check_termination()
        return action

    def get_grid(self):
        """Legacy-friendly integer map for existing visualizations."""
        # 0: empty, 1: macrophage, 2: bacteria, 3: nutrient-rich, 4: blocked, 5: neutrophil
        grid = np.zeros((self.height, self.width), dtype=int)
        for x, y in self.blocked_tiles:
            grid[y, x] = 4

        # Mark nutrient-rich spots for display purposes.
        nutrient_mask = self.nutrients >= config.REPLICATION_REQUIRES_MIN_PATCH_RESOURCE
        grid[nutrient_mask] = np.where(grid[nutrient_mask] == 0, 3, grid[nutrient_mask])

        for n in self.neutrophils:
            x, y = n.position
            if grid[y, x] != 4:
                grid[y, x] = 5

        for b in self.bacteria:
            x, y = b.position
            if grid[y, x] != 4:
                grid[y, x] = 2

        mx, my = self.macrophage.position
        grid[my, mx] = 1
        return grid

    def clone(self):
        env_clone = Environment.__new__(Environment)
        env_clone.rng = random.Random()
        env_clone.rng.setstate(self.rng.getstate())
        env_clone.size = self.size
        env_clone.width = self.width
        env_clone.height = self.height
        env_clone.step_count = self.step_count
        env_clone.done = self.done
        env_clone.winner = self.winner
        env_clone.win_reason = self.win_reason

        env_clone.compartment_map = np.array(self.compartment_map, copy=True)
        env_clone.blocked_tiles = set(self.blocked_tiles)
        env_clone.surface_patches = set(self.surface_patches)
        env_clone.nutrients = np.array(self.nutrients, copy=True)
        env_clone.chemokine = np.array(self.chemokine, copy=True)
        env_clone.recruitment_queue = deque(self.recruitment_queue)

        env_clone.macrophage = Macrophage(
            position=self.macrophage.position,
            health=self.macrophage.health,
            signal_cooldown=self.macrophage.signal_cooldown,
            kills=self.macrophage.kills,
        )

        env_clone.bacteria = [
            Bacteria(
                position=b.position,
                health=b.health,
                age=b.age,
                state=b.state,
                attach_timer=b.attach_timer,
                biofilm_timer=b.biofilm_timer,
                dispersal_cooldown=b.dispersal_cooldown,
            )
            for b in self.bacteria
        ]
        env_clone.neutrophils = [
            Neutrophil(position=n.position, health=n.health, age=n.age) for n in self.neutrophils
        ]

        env_clone.killed_bacteria = self.killed_bacteria
        env_clone.tissue_damage = self.tissue_damage
        env_clone.immune_usage = self.immune_usage
        env_clone.chemokine_events = self.chemokine_events
        env_clone.current_phase = self.current_phase

        env_clone.history = None
        env_clone.macrophage_position_frequency = None
        env_clone.action_effect_frequency = None
        return env_clone

    # ---------- Action and perception ----------

    def get_macrophage_actions(self):
        actions = []
        mx, my = self.macrophage.position
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
            nx, ny = mx + dx, my + dy
            if self._is_walkable((nx, ny)):
                actions.append(("move", (dx, dy)))

        if self._adjacent_bacteria(self.macrophage.position):
            actions.append(("attack", None))

        if self.macrophage.signal_cooldown <= 0:
            actions.append(("signal", None))

        return actions

    def get_local_bacteria(self, center, radius):
        cx, cy = center
        return [
            b
            for b in self.bacteria
            if abs(b.position[0] - cx) + abs(b.position[1] - cy) <= radius
        ]

    def get_local_neutrophils(self, center, radius):
        cx, cy = center
        return [
            n
            for n in self.neutrophils
            if abs(n.position[0] - cx) + abs(n.position[1] - cy) <= radius
        ]

    # ---------- Internal mechanics ----------

    def _execute_macrophage_action(self, action):
        if action is None:
            return

        atype, param = action
        if atype == "move":
            dx, dy = param
            mx, my = self.macrophage.position
            new_pos = (mx + dx, my + dy)
            if self._is_walkable(new_pos):
                self.macrophage.position = new_pos

        elif atype == "attack":
            targets = self._adjacent_bacteria(self.macrophage.position)
            for b in targets:
                damage = config.MACROPHAGE_BASE_KILL_DAMAGE
                if b.state in ("microcolony", "biofilm"):
                    damage = int(round(damage * config.BIOFILM_DEFENSE_MULTIPLIER))
                b.health -= max(1, damage)
            self._remove_dead_bacteria()

            mx, my = self.macrophage.position
            self.chemokine[my, mx] += config.CHEMOKINE_RELEASE_PER_CONTACT
            self.chemokine_events += 1

        elif atype == "signal":
            mx, my = self.macrophage.position
            self.chemokine[my, mx] += config.MACROPHAGE_SIGNAL_STRENGTH
            self.macrophage.signal_cooldown = config.MACROPHAGE_SIGNAL_COOLDOWN
            self.immune_usage += 1.0
            self.chemokine_events += 1

        if self.macrophage.signal_cooldown > 0 and atype != "signal":
            self.macrophage.signal_cooldown -= 1

    def _execute_bacteria_phase(self):
        if not self.bacteria:
            return

        for b in self.bacteria:
            b.age += 1
            if b.dispersal_cooldown > 0:
                b.dispersal_cooldown -= 1

        # Bacteria attack if adjacent to macrophage.
        mx, my = self.macrophage.position
        for b in self.bacteria:
            bx, by = b.position
            if abs(bx - mx) + abs(by - my) <= 1:
                if b.age >= config.BACTERIA_MATURITY_AGE and self.rng.random() < config.BACTERIA_ATTACK_PROB:
                    self.macrophage.health -= config.BACTERIA_ATTACK_DAMAGE
                self.chemokine[my, mx] += config.CHEMOKINE_RELEASE_PER_CONTACT * 0.35

        # Movement and state transitions.
        for b in self.bacteria:
            self._update_bacteria_state(b)
            self._move_bacteria(b)

        # Replication.
        newborns = []
        for b in self.bacteria:
            if self._can_replicate(b) and self.rng.random() < self._replication_probability(b):
                spawn = self._find_spawn_location(b.position)
                if spawn is not None:
                    newborns.append(Bacteria(position=spawn, health=config.BACTERIA_INITIAL_HEALTH))

        self.bacteria.extend(newborns)

    def _execute_neutrophil_phase(self):
        if not self.neutrophils:
            return

        for n in self.neutrophils:
            n.age += 1
            self._move_neutrophil(n)
            self._neutrophil_attack(n)

        self.neutrophils = [n for n in self.neutrophils if n.age < config.NEUTROPHIL_LIFESPAN]

    def _update_fields_and_recruitment(self):
        # Nutrient regeneration.
        active_mask = self.compartment_map >= 0
        self.nutrients[active_mask] = np.minimum(
            config.PATCH_NUTRIENT_CAPACITY,
            self.nutrients[active_mask] + config.PATCH_REGEN_RATE,
        )

        # Nutrient consumption by bacteria.
        for b in self.bacteria:
            x, y = b.position
            consumed = min(self.nutrients[y, x], 1.2)
            self.nutrients[y, x] -= consumed
            b.health = min(config.BACTERIA_MAX_HEALTH, b.health + int(consumed > 0.3))

        # Chemokine diffusion + decay + small noise.
        self._diffuse_chemokine()
        self.chemokine *= config.CHEMOKINE_DECAY

        if config.CHEMOKINE_GRADIENT_NOISE > 0:
            noise = np.random.normal(0.0, config.CHEMOKINE_GRADIENT_NOISE, size=self.chemokine.shape)
            self.chemokine += noise
            self.chemokine = np.clip(self.chemokine, 0.0, None)

        # Recruitment triggers.
        peak = float(self.chemokine.max())
        if peak >= config.NEUTROPHIL_RECRUITMENT_THRESHOLD and len(self.neutrophils) < config.MAX_NEUTROPHIL_POOL:
            due = self.step_count + config.NEUTROPHIL_ARRIVAL_DELAY
            self.recruitment_queue.append(due)

        while self.recruitment_queue and self.recruitment_queue[0] <= self.step_count:
            self.recruitment_queue.popleft()
            if len(self.neutrophils) < config.MAX_NEUTROPHIL_POOL:
                self._spawn_neutrophil()
                self.immune_usage += 0.5

        # Tissue damage model.
        self.tissue_damage += len(self.neutrophils) * config.TISSUE_DAMAGE_FROM_NEUTROPHILS
        self.tissue_damage += len(self.bacteria) * config.TISSUE_DAMAGE_FROM_BACTERIA
        if len(self.neutrophils) > 0 and len(self.bacteria) > 0:
            self.tissue_damage += config.DAMAGE_FROM_PROLONGED_INFLAMMATION

    def _diffuse_chemokine(self):
        new_field = np.array(self.chemokine, copy=True)
        for y in range(self.height):
            for x in range(self.width):
                if not self._is_walkable((x, y)):
                    continue
                share = self.chemokine[y, x] * config.CHEMOKINE_DIFFUSION_RATE
                if share <= 0:
                    continue
                neighbors = self._valid_neighbors((x, y), include_stay=False)
                if not neighbors:
                    continue
                per_n = share / len(neighbors)
                new_field[y, x] -= share
                for nx, ny in neighbors:
                    new_field[ny, nx] += per_n
        self.chemokine = np.clip(new_field, 0.0, None)

    def _update_bacteria_state(self, b):
        x, y = b.position
        local_chemokine = self.chemokine[y, x]
        on_surface = (x, y) in self.surface_patches

        if b.state == "planktonic":
            if on_surface and self.nutrients[y, x] >= config.REPLICATION_REQUIRES_MIN_PATCH_RESOURCE:
                b.attach_timer += 1
            else:
                b.attach_timer = 0
            if b.attach_timer >= config.ATTACHMENT_TIME_REQUIRED:
                b.state = "attached"
                b.attach_timer = 0

        elif b.state == "attached":
            if local_chemokine > config.NEUTROPHIL_RECRUITMENT_THRESHOLD * 0.8 and b.dispersal_cooldown == 0:
                b.state = "planktonic"
                b.dispersal_cooldown = config.BIOFILM_DISPERSAL_DELAY
            else:
                b.biofilm_timer += 1
                if b.biofilm_timer >= config.BIOFILM_BUILD_TIME:
                    b.state = "microcolony"

        elif b.state == "microcolony":
            if b.biofilm_timer >= config.BIOFILM_BUILD_TIME + 2:
                b.state = "biofilm"
            b.biofilm_timer += 1

        elif b.state == "biofilm":
            if local_chemokine > config.NEUTROPHIL_RECRUITMENT_THRESHOLD * 1.2 and b.dispersal_cooldown == 0:
                b.state = "planktonic"
                b.biofilm_timer = 0
                b.dispersal_cooldown = config.BIOFILM_DISPERSAL_DELAY

    def _move_bacteria(self, b):
        # Biofilm and microcolonies do not move.
        if config.BIOFILM_MOTILITY_PENALTY and b.state in ("microcolony", "biofilm"):
            return

        from agents.bacteria_adaptive import choose_bacteria_move

        dx, dy = choose_bacteria_move(b, self)
        nx, ny = b.position[0] + dx, b.position[1] + dy
        next_pos = (nx, ny)

        if not self._is_walkable(next_pos):
            return
        if next_pos == self.macrophage.position:
            return
        if any(other is not b and other.position == next_pos for other in self.bacteria):
            return

        b.position = next_pos

    def _can_replicate(self, b):
        x, y = b.position
        if config.REPLICATION_REQUIRES_ATTACHMENT and b.state == "planktonic":
            return False
        if self.nutrients[y, x] < config.REPLICATION_REQUIRES_MIN_PATCH_RESOURCE:
            return False
        if self._local_bacteria_count((x, y), radius=1) >= config.LOCAL_CARRYING_CAPACITY:
            return False
        return True

    def _replication_probability(self, b):
        x, y = b.position
        chemokine_pressure = self.chemokine[y, x] * config.IMMUNE_PRESSURE_REPLICATION_PENALTY
        p = config.BASE_REPLICATION_PROB - chemokine_pressure
        if b.state in ("microcolony", "biofilm"):
            p *= config.BIOFILM_REPLICATION_MODIFIER
        return max(0.01, min(0.95, p))

    def _find_spawn_location(self, center):
        neighbors = self._valid_neighbors(center, include_stay=False)
        self.rng.shuffle(neighbors)
        for pos in neighbors:
            if pos == self.macrophage.position:
                continue
            if any(b.position == pos for b in self.bacteria):
                continue
            return pos
        return None

    def _move_neutrophil(self, n):
        nearby = self.get_local_bacteria(n.position, config.NEUTROPHIL_SENSE_RADIUS)
        if nearby:
            target = min(nearby, key=lambda b: self._manhattan(n.position, b.position)).position
            dx, dy = self._step_toward(n.position, target)
        else:
            dx, dy = self.rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)])

        next_pos = (n.position[0] + dx, n.position[1] + dy)
        if self._is_walkable(next_pos) and next_pos != self.macrophage.position:
            n.position = next_pos

    def _neutrophil_attack(self, n):
        targets = self._adjacent_bacteria(n.position)
        for b in targets:
            b.health -= config.NEUTROPHIL_KILL_DAMAGE
            bx, by = b.position
            if self.action_effect_frequency is not None:
                self.action_effect_frequency[by, bx] += 1
        self._remove_dead_bacteria()

    def _remove_dead_bacteria(self):
        survivors = []
        for b in self.bacteria:
            if b.health > 0:
                survivors.append(b)
            else:
                self.killed_bacteria += 1
                self.macrophage.kills += 1
        self.bacteria = survivors

    def _spawn_neutrophil(self):
        # Spawn near boundaries to mimic recruited entry.
        edge_cells = []
        for x in range(self.width):
            edge_cells.extend([(x, 0), (x, self.height - 1)])
        for y in range(self.height):
            edge_cells.extend([(0, y), (self.width - 1, y)])

        self.rng.shuffle(edge_cells)
        for pos in edge_cells:
            if not self._is_walkable(pos):
                continue
            if pos == self.macrophage.position:
                continue
            if any(b.position == pos for b in self.bacteria):
                continue
            self.neutrophils.append(Neutrophil(position=pos, health=40))
            return

    def _update_phase_label(self):
        if len(self.neutrophils) > 0 and len(self.bacteria) > 0:
            self.current_phase = "escalation"
        elif self.recruitment_queue:
            self.current_phase = "recruitment"
        elif self.chemokine.max() > 1.0:
            self.current_phase = "detection_containment"
        else:
            self.current_phase = "seeding"

    def _check_termination(self):
        if self.macrophage.health <= 0:
            self.done = True
            self.winner = "Bacteria"
            self.win_reason = "macrophage died"
            return

        if self.tissue_damage >= config.TISSUE_DAMAGE_FAIL_THRESHOLD:
            self.done = True
            self.winner = "Bacteria"
            self.win_reason = "host damage threshold exceeded"
            return

        if len(self.bacteria) == 0:
            self.done = True
            self.winner = "Host"
            self.win_reason = "all bacteria cleared"
            return

        if self.step_count >= config.TIME_HORIZON:
            self.done = True
            self.winner = "Bacteria"
            self.win_reason = "time horizon reached with persistence"

    def _record_history(self):
        if self.history is None:
            return

        self.history["step"].append(self.step_count)
        self.history["phase"].append(self.current_phase)
        self.history["macrophage_health"].append(self.macrophage.health)
        self.history["bacteria_count"].append(len(self.bacteria))
        self.history["neutrophil_count"].append(len(self.neutrophils))
        self.history["tissue_damage"].append(float(self.tissue_damage))
        self.history["host_utility"].append(self.compute_host_utility())
        self.history["colonized_compartments"].append(self.colonized_compartment_count())
        self.history["chemokine_peak"].append(float(self.chemokine.max()))
        self.history["nutrient_total"].append(float(self.nutrients.sum()))

    # ---------- Utility and diagnostics ----------

    def colonized_compartment_count(self):
        occupied = set()
        for b in self.bacteria:
            x, y = b.position
            cid = int(self.compartment_map[y, x])
            if cid >= 0:
                occupied.add(cid)
        return len(occupied)

    def compute_host_utility(self):
        return (
            float(self.killed_bacteria)
            - config.HOST_UTILITY_ALPHA * float(self.tissue_damage)
            - config.HOST_UTILITY_BETA * float(self.immune_usage)
        )

    # ---------- Spatial helpers ----------

    def _is_walkable(self, pos):
        x, y = pos
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        if (x, y) in self.blocked_tiles:
            return False
        return self.compartment_map[y, x] >= 0

    def _valid_neighbors(self, pos, include_stay=False):
        x, y = pos
        candidates = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if include_stay:
            candidates.append((0, 0))
        out = []
        for dx, dy in candidates:
            nxt = (x + dx, y + dy)
            if self._is_walkable(nxt):
                out.append(nxt)
        return out

    def _adjacent_bacteria(self, pos):
        x, y = pos
        return [
            b
            for b in self.bacteria
            if abs(b.position[0] - x) + abs(b.position[1] - y) <= config.MACROPHAGE_ATTACK_RADIUS
        ]

    def _local_bacteria_count(self, center, radius):
        cx, cy = center
        return sum(
            1 for b in self.bacteria if abs(b.position[0] - cx) + abs(b.position[1] - cy) <= radius
        )

    @staticmethod
    def _manhattan(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _step_toward(self, src, target):
        sx, sy = src
        tx, ty = target
        dx = 0
        dy = 0
        if sx < tx:
            dx = 1
        elif sx > tx:
            dx = -1
        if sy < ty:
            dy = 1
        elif sy > ty:
            dy = -1

        options = [(dx, 0), (0, dy), (dx, dy), (0, 0)]
        for ox, oy in options:
            if self._is_walkable((sx + ox, sy + oy)):
                return ox, oy
        return 0, 0

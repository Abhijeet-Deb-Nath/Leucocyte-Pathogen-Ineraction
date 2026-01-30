from dataclasses import dataclass

@dataclass
class Macrophage:
    position: tuple  # (x, y) coordinates on the grid
    health: int
    exhaustion: int = 0  # Exhaustion level from fighting multiple bacteria
    last_toxin_step: int = -999  # Last step when toxin was used (for cooldown)

@dataclass
class Bacteria:
    position: tuple  # (x, y) coordinates on the grid
    health: int
    nutrient_boost: bool = False  # whether this bacteria has a nutrient boost for replication chance
    age: int = 0  # Age in simulation steps (newborns are immature)
    in_biofilm: bool = False  # Whether part of a biofilm cluster
    virulence_active: bool = False  # Whether actively producing virulence factors

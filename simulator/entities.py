from dataclasses import dataclass, field
from typing import Literal

BacteriaState = Literal["planktonic", "attached", "microcolony", "biofilm"]


@dataclass
class Macrophage:
    position: tuple[int, int]
    health: int
    signal_cooldown: int = 0
    kills: int = 0


@dataclass
class Bacteria:
    position: tuple[int, int]
    health: int
    age: int = 0
    state: BacteriaState = "planktonic"
    attach_timer: int = 0
    biofilm_timer: int = 0
    dispersal_cooldown: int = 0


@dataclass
class Neutrophil:
    position: tuple[int, int]
    health: int
    age: int = 0
    recent_positions: list[tuple[int, int]] = field(default_factory=list)

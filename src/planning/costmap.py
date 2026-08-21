import math
from typing import List, Tuple


class Costmap:
    """A sparse obstacle map: a set of inflated circles in world coordinates."""

    def __init__(self, inflation_radius: float = 0.15):
        self.inflation_radius = inflation_radius
        self._obstacles: List[Tuple[float, float]] = []

    def add_obstacle(self, x: float, y: float) -> None:
        self._obstacles.append((x, y))

    def is_occupied(self, x: float, y: float) -> bool:
        return any(
            math.hypot(x - ox, y - oy) <= self.inflation_radius
            for ox, oy in self._obstacles
        )

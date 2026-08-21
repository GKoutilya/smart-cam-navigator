import heapq
import math
from typing import List, Optional, Tuple

from src.planning.costmap import Costmap

Point = Tuple[float, float]
Cell = Tuple[int, int]

_NEIGHBOR_OFFSETS = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
]

_RESCUE_SEARCH_RINGS = 5


def _to_cell(point: Point, origin: Point, resolution: float) -> Cell:
    return (round((point[0] - origin[0]) / resolution), round((point[1] - origin[1]) / resolution))


def _to_world(cell: Cell, origin: Point, resolution: float) -> Point:
    return (origin[0] + cell[0] * resolution, origin[1] + cell[1] * resolution)


def _nearest_free_cell(cell: Cell, costmap: Costmap, origin: Point, resolution: float) -> Optional[Cell]:
    if not costmap.is_occupied(*_to_world(cell, origin, resolution)):
        return cell

    for radius in range(1, _RESCUE_SEARCH_RINGS + 1):
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                candidate = (cell[0] + dx, cell[1] + dy)
                if not costmap.is_occupied(*_to_world(candidate, origin, resolution)):
                    return candidate
    return None


def find_path(start: Point, goal: Point, costmap: Costmap,
              resolution: float = 0.05, padding: float = 0.5) -> List[Point]:
    """8-connected A* over a grid built just large enough to bound start/goal.

    Raises ValueError if start/goal are blocked with no nearby free cell, or
    if no path exists between them.
    """
    origin = (min(start[0], goal[0]) - padding, min(start[1], goal[1]) - padding)

    start_cell = _nearest_free_cell(_to_cell(start, origin, resolution), costmap, origin, resolution)
    goal_cell = _nearest_free_cell(_to_cell(goal, origin, resolution), costmap, origin, resolution)
    if start_cell is None or goal_cell is None:
        raise ValueError("Start or goal is blocked and no free cell was found nearby.")

    path_cells = _astar_search(start_cell, goal_cell, costmap, origin, resolution)
    if path_cells is None:
        raise ValueError("No feasible path found between start and goal.")

    return [start] + [_to_world(cell, origin, resolution) for cell in path_cells] + [goal]


def _astar_search(start_cell: Cell, goal_cell: Cell, costmap: Costmap,
                   origin: Point, resolution: float) -> Optional[List[Cell]]:
    open_heap = [(0.0, start_cell)]
    came_from = {}
    g_score = {start_cell: 0.0}
    visited = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal_cell:
            return _reconstruct_path(came_from, current)
        if current in visited:
            continue
        visited.add(current)

        for dx, dy in _NEIGHBOR_OFFSETS:
            neighbor = (current[0] + dx, current[1] + dy)
            if neighbor in visited or costmap.is_occupied(*_to_world(neighbor, origin, resolution)):
                continue

            tentative_g = g_score[current] + math.hypot(dx, dy) * resolution
            if tentative_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative_g
                came_from[neighbor] = current
                f_score = tentative_g + _heuristic(neighbor, goal_cell, resolution)
                heapq.heappush(open_heap, (f_score, neighbor))

    return None


def _heuristic(cell: Cell, goal_cell: Cell, resolution: float) -> float:
    return math.hypot(cell[0] - goal_cell[0], cell[1] - goal_cell[1]) * resolution


def _reconstruct_path(came_from: dict, current: Cell) -> List[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

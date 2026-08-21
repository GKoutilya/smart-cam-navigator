import math
from typing import List, Tuple


class PathOptimizer:
    def __init__(self):
        pass

    def time_parameterize(self, path: List[Tuple[float, float]],
                           max_velocity: float = 0.5,
                           max_acceleration: float = 0.5) -> List[Tuple[float, float, float]]:
        """
            Time-parameterizes a path with a trapezoidal velocity profile
            (triangular if the path is too short to reach max_velocity),
            returning (t, x, y) waypoints that respect max_velocity/
            max_acceleration by construction rather than checked after the fact.
        """
        if max_velocity <= 0 or max_acceleration <= 0:
            raise ValueError("max_velocity and max_acceleration must be positive.")
        if not path:
            return []

        cumulative_distances = [0.0]
        for (x0, y0), (x1, y1) in zip(path, path[1:]):
            cumulative_distances.append(cumulative_distances[-1] + math.hypot(x1 - x0, y1 - y0))
        total_distance = cumulative_distances[-1]

        accel_distance = min(max_velocity ** 2 / (2 * max_acceleration), total_distance / 2)
        peak_velocity = math.sqrt(2 * max_acceleration * accel_distance)
        cruise_distance = max(total_distance - 2 * accel_distance, 0.0)
        accel_time = peak_velocity / max_acceleration
        cruise_time = cruise_distance / peak_velocity if peak_velocity > 0 else 0.0
        total_time = 2 * accel_time + cruise_time

        def time_at(distance):
            if distance <= accel_distance:
                return math.sqrt(2 * distance / max_acceleration)
            if distance <= accel_distance + cruise_distance:
                return accel_time + (distance - accel_distance) / peak_velocity
            remaining = total_distance - distance
            return total_time - math.sqrt(2 * remaining / max_acceleration)

        return [(time_at(distance), x, y) for distance, (x, y) in zip(cumulative_distances, path)]

    def optimize_path(self, path, max_velocity: float = 0.5, max_acceleration: float = 0.5):
        return self.time_parameterize(path, max_velocity=max_velocity, max_acceleration=max_acceleration)

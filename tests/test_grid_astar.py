import math
import unittest

from src.planning.costmap import Costmap
from src.planning.grid_astar import find_path


class TestGridAstar(unittest.TestCase):
    def test_open_space_path_is_near_straight_line(self):
        start = (0.0, 0.0)
        goal = (1.0, 1.0)
        path = find_path(start, goal, Costmap())

        straight_line_distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
        path_length = sum(
            math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(path, path[1:])
        )
        self.assertAlmostEqual(path_length, straight_line_distance, places=1)

    def test_path_detours_around_obstacle(self):
        costmap = Costmap(inflation_radius=0.2)
        costmap.add_obstacle(0.5, 0.5)  # sits directly on the straight line start->goal

        path = find_path((0.0, 0.5), (1.0, 0.5), costmap)

        for x, y in path:
            self.assertGreater(math.hypot(x - 0.5, y - 0.5), 0.2 - 1e-6)

        straight_line_distance = 1.0
        path_length = sum(
            math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(path, path[1:])
        )
        self.assertGreater(path_length, straight_line_distance)

    def test_raises_when_goal_is_fully_enclosed(self):
        costmap = Costmap(inflation_radius=0.5)
        # Ring of obstacles tightly enclosing the goal so no free cell exists nearby.
        for angle_deg in range(0, 360, 30):
            angle = math.radians(angle_deg)
            costmap.add_obstacle(5.0 + 0.4 * math.cos(angle), 5.0 + 0.4 * math.sin(angle))

        with self.assertRaises(ValueError):
            find_path((0.0, 0.0), (5.0, 5.0), costmap, resolution=0.05)


if __name__ == '__main__':
    unittest.main()

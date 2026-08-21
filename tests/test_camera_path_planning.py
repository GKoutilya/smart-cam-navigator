import math
import unittest

from src.planning.camera_path_planning import CameraPathPlanner
from src.planning.costmap import Costmap


class TestCameraPathPlanner(unittest.TestCase):

    def setUp(self):
        self.planner = CameraPathPlanner()

    def test_plan_path_reaches_goal_in_open_space(self):
        start = (0.0, 0.0)
        goals = [(1.0, 1.0)]
        path = self.planner.plan_path(start, goals)

        self.assertGreater(len(path), 0)
        self.assertEqual(path[0], start)
        self.assertAlmostEqual(path[-1][0], goals[0][0], places=1)
        self.assertAlmostEqual(path[-1][1], goals[0][1], places=1)

    def test_plan_path_through_multiple_goals(self):
        start = (0.0, 0.0)
        goals = [(1.0, 0.0), (1.0, 1.0)]
        path = self.planner.plan_path(start, goals)

        self.assertAlmostEqual(path[-1][0], goals[-1][0], places=1)
        self.assertAlmostEqual(path[-1][1], goals[-1][1], places=1)

    def test_plan_path_detours_around_obstacle(self):
        costmap = Costmap(inflation_radius=0.2)
        costmap.add_obstacle(0.5, 0.5)  # sits directly between start and goal
        self.planner.set_costmap(costmap)

        start = (0.0, 0.5)
        goal = (1.0, 0.5)
        path = self.planner.plan_path(start, [goal])

        for x, y in path:
            self.assertGreater(math.hypot(x - 0.5, y - 0.5), 0.2 - 1e-6)

        straight_line_distance = math.hypot(goal[0] - start[0], goal[1] - start[1])
        path_length = sum(
            math.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(path, path[1:])
        )
        self.assertGreater(path_length, straight_line_distance)


if __name__ == '__main__':
    unittest.main()

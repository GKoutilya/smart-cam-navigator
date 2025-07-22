import unittest
from src.planning.camera_path_planning import CameraPathPlanner

class TestCameraPathPlanner(unittest.TestCase):

    def setUp(self):
        self.planner = CameraPathPlanner()

    def test_plan_path(self):
        start = (0, 0)
        goal = (10, 10)
        path = self.planner.plan_path(start, goal)
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)

    def test_optimize_path(self):
        path = [(0, 0), (5, 5), (10, 10)]
        optimized_path = self.planner.optimize_path(path)
        self.assertIsNotNone(optimized_path)
        self.assertLess(len(optimized_path), len(path))

if __name__ == '__main__':
    unittest.main()
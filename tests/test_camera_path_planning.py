import unittest
from src.planning.camera_path_planning import CameraPathPlanner

class TestCameraPathPlanner(unittest.TestCase):

    def setUp(self):
        self.planner = CameraPathPlanner(image_width=1920, image_height=1080)

    def test_plan_path(self):
        start = (0, 0)
        goal = (10, 10)
        path = self.planner.plan_path(start, goal)
        self.assertIsNotNone(path)
        self.assertGreater(len(path), 0)

    def test_optimize_path(self):
        path = [(0, 0), (3, 4), (6, 8)]
        optimized_path = self.planner.optimize_path(path)
        self.assertIsNotNone(optimized_path)
        self.assertEqual(len(optimized_path), len(path))
        self.assertTrue(any(
            abs(x1 - x2) < 1e-6 or abs(y1 - y2) > 1e-6
            for (x1, y1), (x2, y2) in zip(path, optimized_path)
        ))


if __name__ == '__main__':
    unittest.main()
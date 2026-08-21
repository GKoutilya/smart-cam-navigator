import unittest

from src.planning.path_optimizer import PathOptimizer


class TestPathOptimizer(unittest.TestCase):
    def setUp(self):
        self.optimizer = PathOptimizer()

    def test_empty_path(self):
        self.assertEqual(self.optimizer.time_parameterize([]), [])

    def test_single_point_path(self):
        result = self.optimizer.time_parameterize([(1.0, 2.0)])
        self.assertEqual(result, [(0.0, 1.0, 2.0)])

    def test_time_strictly_increasing(self):
        path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (5.0, 0.0)]
        result = self.optimizer.time_parameterize(path, max_velocity=1.0, max_acceleration=1.0)

        times = [t for t, _, _ in result]
        self.assertEqual(times, sorted(times))
        self.assertEqual(times[0], 0.0)

    def test_trapezoidal_profile_total_time(self):
        # 10-unit path, v_max=1, a_max=1: accelerate 1s (0.5 units), cruise 9s
        # (9 units) at 1 unit/s, decelerate 1s (0.5 units) -> 11s total.
        path = [(0.0, 0.0), (10.0, 0.0)]
        result = self.optimizer.time_parameterize(path, max_velocity=1.0, max_acceleration=1.0)
        self.assertAlmostEqual(result[-1][0], 11.0, places=3)

    def test_triangular_profile_for_short_path(self):
        # 1-unit path with a high max_velocity never actually reached: pure
        # accelerate-then-decelerate over a_max=1 -> total time 2*sqrt(1/1)=2s.
        path = [(0.0, 0.0), (1.0, 0.0)]
        result = self.optimizer.time_parameterize(path, max_velocity=10.0, max_acceleration=1.0)
        self.assertAlmostEqual(result[-1][0], 2.0, places=3)

    def test_rejects_non_positive_limits(self):
        with self.assertRaises(ValueError):
            self.optimizer.time_parameterize([(0.0, 0.0), (1.0, 0.0)], max_velocity=0)
        with self.assertRaises(ValueError):
            self.optimizer.time_parameterize([(0.0, 0.0), (1.0, 0.0)], max_acceleration=0)


if __name__ == '__main__':
    unittest.main()

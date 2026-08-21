import unittest

from src.planning.costmap import Costmap


class TestCostmap(unittest.TestCase):
    def test_no_obstacles_nothing_occupied(self):
        costmap = Costmap()
        self.assertFalse(costmap.is_occupied(0.0, 0.0))

    def test_point_within_inflation_radius_is_occupied(self):
        costmap = Costmap(inflation_radius=0.2)
        costmap.add_obstacle(1.0, 1.0)

        self.assertTrue(costmap.is_occupied(1.1, 1.0))  # 0.1 away, within radius

    def test_point_outside_inflation_radius_is_free(self):
        costmap = Costmap(inflation_radius=0.2)
        costmap.add_obstacle(1.0, 1.0)

        self.assertFalse(costmap.is_occupied(2.0, 2.0))


if __name__ == '__main__':
    unittest.main()

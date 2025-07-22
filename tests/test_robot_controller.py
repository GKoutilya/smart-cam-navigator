import unittest
from src.action.robot_controller import RobotController

class TestRobotController(unittest.TestCase):

    def setUp(self):
        self.controller = RobotController()

    def test_execute_path(self):
        path = [(0, 0), (5, 5), (10, 10)]
        result = self.controller.execute_path(path)
        self.assertTrue(result)

    def test_interact_with_hardware(self):
        command = "move_to"
        parameters = {"position": (5, 5)}
        result = self.controller.interact_with_hardware(command, parameters)
        self.assertTrue(result)

if __name__ == '__main__':
    unittest.main()
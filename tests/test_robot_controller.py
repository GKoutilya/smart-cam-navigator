import unittest
from src.action.robot_controller import RobotController

class MockCamera:
    def capture_image(self):
        return "mock_image"
    
class MockVision:
    def __init__(self):
        self.camera = MockCamera()

    def capture_image(self):
        return self.camera.capture_image()
    
    def process_image(self, image):
        return {"objects": ["mock_object"], "layout": "mock_layout"}

class TestRobotController(unittest.TestCase):
    def setUp(self):
        self.vision = MockVision()
        self.path = [(0,0), (5,5), (10,10)]
        self.controller = RobotController(vision=self.vision)

    def test_execute_path(self):
        result = self.controller.execute_path(self.path)
        self.assertTrue(result)
        self.assertEqual(len(self.controller.image_captured), len(self.path))
        for entry in self.controller.image_captured:
            self.assertIn("mock_image", entry["image"])
            self.assertIn("mock_object", entry["scene_info"]["objects"])

    def test_status(self):
        status = self.controller.get_status()
        self.assertEqual(status, "Robot is operational.")


class FakeRobot:
    """A lightweight double for SimulatedRobot - tests RobotController's
    delegation logic without depending on real Pymunk physics."""

    def __init__(self):
        self.commanded = []
        self.stepped_dt = None

    def command_towards(self, x, y, dt):
        self.commanded.append((x, y, dt))

    def step(self, dt):
        self.stepped_dt = dt

    def get_position(self):
        return (1.5, 2.5)


class TestRobotControllerMoveTo(unittest.TestCase):
    def test_move_to_delegates_to_robot_when_present(self):
        fake_robot = FakeRobot()
        controller = RobotController(vision=MockVision(), robot=fake_robot)

        controller.move_to((3.0, 4.0), dt=0.05)

        self.assertEqual(fake_robot.commanded, [(3.0, 4.0, 0.05)])
        self.assertEqual(fake_robot.stepped_dt, 0.05)
        self.assertEqual(controller.current_position, (1.5, 2.5))

    def test_move_to_falls_back_to_teleport_without_a_robot(self):
        controller = RobotController(vision=MockVision())

        controller.move_to((3.0, 4.0))

        self.assertEqual(controller.current_position, (3.0, 4.0))


if __name__ == '__main__':
    unittest.main()
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

if __name__ == '__main__':
    unittest.main()
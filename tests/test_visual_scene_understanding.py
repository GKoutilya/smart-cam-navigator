import cv2
import unittest
from unittest.mock import Mock
import pillow_heif
import numpy as np
from PIL import Image
from ultralytics import YOLO
from src.perception.visual_scene_understanding import VisualSceneUnderstanding

heif_file = pillow_heif.read_heif(r"C:\Users\kouti\Downloads\IMG_4525.HEIC")

image_temp = Image.frombytes(
    heif_file.mode, 
    heif_file.size, 
    heif_file.data
)

image = cv2.cvtColor(np.array(image_temp), cv2.COLOR_RGB2BGR)

class TestVisualSceneUnderstanding(unittest.TestCase):

    def setUp(self):
        model = YOLO('yolov8n.pt')
        self.understanding = VisualSceneUnderstanding(detection_model=model, camera=Mock())

    def test_pose_estimation(self):
        pose = self.understanding.estimate_pose(image)
        self.assertIsNotNone(pose)
        self.assertEqual(len(pose), 3)  # Assuming pose is represented by 6 parameters (x, y, z, roll, pitch, yaw)

    def test_scene_classification(self):
        scene_type = self.understanding.classify_scene(image)
        self.assertIsNotNone(scene_type)
        self.assertIn(scene_type, ['indoor', 'outdoor', 'urban', 'rural'])  # Example scene types
    
    def test_visualize_people_detection(self):
        self.assertIsNotNone(image, "Test image not found or unable to load.")

        results = self.understanding.process_image(image)

        for person in results["people"]['detections']:
            x1, y1, x2, y2 = person["bbox"]
            confidence = person["confidence"]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"Person: {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("People Detection", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    unittest.main()
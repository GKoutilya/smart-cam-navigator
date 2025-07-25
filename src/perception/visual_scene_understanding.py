from  ultralytics import YOLO
import cv2
import numpy as np
import random
from typing import Any, Dict

class VisualSceneUnderstanding:
    def __init__(self, detection_model, camera, pose_model=None, scene_classifier=None):
        self.camera = camera
        self.pose_model = pose_model
        if detection_model is None:
            self.detection_model = YOLO('yolov8n.pt')
        else:
            self.detection_model = detection_model
        self.scene_classifier = scene_classifier

    def estimate_pose(self, image: Any) -> Dict[str, float]:
        # Dummy pose vector
        return {
            "x": round(random.uniform(0, 10), 2),
            "y": round(random.uniform(0, 10), 2),
            "theta": round(random.uniform(-3.14, 3.14), 2)
        }
    
    def capture_image(self):
        return self.camera.capture_image()

    def detect_people(self, image: Any, conf_threshold=0.5) -> Dict[str, Any]:
        # Convert image if it's a NumPy array (BGR) or assume it's a filepath
        if isinstance(image, str):
            image = cv2.imread(image)

        results = self.detection_model(image)[0] # YOLO returns a list, we want the first result

        people = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if self.detection_model.names[cls_id] == "person" and conf >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                people.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(float(box.conf[0]), 2)
                })

        return {
            "detections": people,
            "num_people": len(people)
        }

    def classify_scene(self, image: Any) -> str:
        # Returns a simple tag of the scene
        return random.choice(["indoor", "outdoor", "urban", "rural"])

    def process_image(self, image: Any) -> Dict[str, Any]:
        pose = self.estimate_pose(image)
        people = self.detect_people(image)
        scene_type = self.classify_scene(image)

        return {
            "pose": pose,
            "people": people,
            "scene_type": scene_type
        }
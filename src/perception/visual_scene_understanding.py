from  ultralytics import YOLO
import cv2
import numpy as np
import random
import torch
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

    def estimate_pose(self) -> Dict[str, float]:
        return {"x": 150, "y": 200, "theta": 0.5}
    
    def capture_image(self):
        return self.camera.capture()

    def detect_people(self, conf_threshold=0.5) -> Dict[str, Any]:
        # Convert image if it's a NumPy array (BGR) or assume it's a filepath
        image = self.camera.capture()

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

    def classify_scene(self) -> str:
        """Returns a simple tag of the scene"""
        image = self.camera.capture()
        results = self.detection_model(image, verbose=False)[0]
        labels = [self.detection_model.model.names[int(cls)] for cls in results.boxes.cls.cpu().numpy()]
        label_set = set(labels)

        if any(obj in label_set for obj in ["oven", "sink", "couch"]):
            return "indoor"
        elif any(obj in label_set for obj in ["car", "bus", "traffic light"]):
            return "urban"
        elif any(obj in label_set for obj in ["tree", "grass", "dog"]):
            return "outdoor"
        elif any(obj in label_set for obj in ["cow", "sheep", "field"]):
            return "rural"
        else:
            return "unknown"

    def process_image(self) -> Dict[str, Any]:
        pose = self.estimate_pose()
        people = self.detect_people()
        scene_type = self.classify_scene()

        return {
            "pose": pose,
            "people": people["detections"],
            "num_people": people["num_people"],
            "scene_type": scene_type
        }
    
    def process_image_pose_only(self):
        image = self.camera.capture()
        if image is None:
            return {"pose": {"x": 0, "y": 0, "theta": 0.0}}  # fail-safe
        pose = self.estimate_pose(image)
        return {"pose": pose}
    
    def infer_goals(self, scene_type: str, image_width, image_height) -> list[tuple[int, int]]:
        results = self.detection_model(self.camera.capture(), verbose=False)[0]
        boxes = results.boxes
        class_ids = boxes.cls.cpu().numpy()
        names = self.detection_model.model.names

        goals = []

        for i, cls_id in enumerate(class_ids):
            label = names[int(cls_id)]
            if label in ["door", "chair", "sofa"]:
                box = boxes.xyxy[i].cpu().numpy()
                x_center = int((box[0] + box[2]) / 2)
                y_center = int((box[1] + box[3]) / 2)
                goals.append((x_center, y_center))

        if not goals:
            goals = [(image_width // 2, image_height // 2)]

        return goals
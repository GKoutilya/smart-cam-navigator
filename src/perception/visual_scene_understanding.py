from ultralytics import YOLO
from typing import Any, Dict, List, Tuple

class VisualSceneUnderstanding:
    def __init__(self, detection_model, camera, pose_model=None, scene_classifier=None):
        self.camera = camera
        self.pose_model = pose_model
        if detection_model is None:
            self.detection_model = YOLO('yolov8n.pt')
        else:
            self.detection_model = detection_model
        self.scene_classifier = scene_classifier

    def capture_image(self):
        return self.camera.capture()

    def _run_detection(self, image):
        return self.detection_model(image, verbose=False)[0]

    def detect_people(self, image=None, results=None, conf_threshold=0.5) -> Dict[str, Any]:
        if results is None:
            if image is None:
                image = self.camera.capture()
            results = self._run_detection(image)

        people = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if self.detection_model.names[cls_id] == "person" and conf >= conf_threshold:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                people.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(conf, 2)
                })

        return {
            "detections": people,
            "num_people": len(people)
        }

    def classify_scene(self, image=None, results=None) -> str:
        """Returns a simple tag of the scene"""
        if results is None:
            if image is None:
                image = self.camera.capture()
            results = self._run_detection(image)

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

    def infer_goals(self, image_width, image_height, image=None, results=None) -> List[Tuple[int, int]]:
        if results is None:
            if image is None:
                image = self.camera.capture()
            results = self._run_detection(image)

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

    def process_image(self, image=None) -> Dict[str, Any]:
        if image is None:
            image = self.camera.capture()

        height, width = image.shape[:2]
        results = self._run_detection(image)
        people = self.detect_people(results=results)
        scene_type = self.classify_scene(results=results)
        goals = self.infer_goals(width, height, results=results)

        return {
            "image": image,
            "people": people["detections"],
            "num_people": people["num_people"],
            "scene_type": scene_type,
            "goals": goals,
        }

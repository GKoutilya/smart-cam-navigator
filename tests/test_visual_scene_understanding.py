import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.perception.visual_scene_understanding import VisualSceneUnderstanding

# Minimal subset of COCO class names actually referenced by classify_scene/infer_goals.
FAKE_NAMES = {0: "person", 1: "couch", 2: "car", 3: "dog", 4: "chair"}


class FakeBoxes:
    """Mimics the subset of ultralytics' Boxes API that VisualSceneUnderstanding relies on."""

    def __init__(self, entries, track_ids=None):
        # entries: list of (class_id, confidence, [x1, y1, x2, y2])
        # track_ids: list of ints, same length as entries, or None if tracking assigned no IDs
        self.cls = torch.tensor([e[0] for e in entries], dtype=torch.float32)
        self.conf = torch.tensor([e[1] for e in entries], dtype=torch.float32)
        self.xyxy = torch.tensor([e[2] for e in entries], dtype=torch.float32)
        self.id = torch.tensor(track_ids, dtype=torch.float32) if track_ids is not None else None

    def __iter__(self):
        for i in range(len(self.cls)):
            box_id = self.id[i:i + 1] if self.id is not None else None
            yield SimpleNamespace(cls=self.cls[i:i + 1], conf=self.conf[i:i + 1], xyxy=self.xyxy[i:i + 1], id=box_id)

    def __len__(self):
        return len(self.cls)


def fake_results(entries, track_ids=None):
    return SimpleNamespace(boxes=FakeBoxes(entries, track_ids=track_ids))


class FakeDetectionModel:
    """Stands in for a loaded YOLO model: its .track() (like YOLO.track(image)) plus .names/.model.names lookups."""

    def __init__(self, entries, names=FAKE_NAMES, track_ids=None):
        self.names = names
        self.model = SimpleNamespace(names=names)
        self._entries = entries
        self._track_ids = track_ids

    def track(self, image, persist=True, verbose=False):
        return [fake_results(self._entries, track_ids=self._track_ids)]


class TestVisualSceneUnderstanding(unittest.TestCase):
    def setUp(self):
        self.understanding = VisualSceneUnderstanding(
            detection_model=SimpleNamespace(names=FAKE_NAMES, model=SimpleNamespace(names=FAKE_NAMES)),
            camera=None,
        )

    def test_detect_people_filters_by_label_and_confidence(self):
        entries = [
            (0, 0.9, [10, 10, 50, 50]),  # person, above threshold
            (0, 0.3, [60, 60, 90, 90]),  # person, below threshold
            (1, 0.9, [0, 0, 20, 20]),    # couch, not a person
        ]
        result = self.understanding.detect_people(results=fake_results(entries), conf_threshold=0.5)

        self.assertEqual(result["num_people"], 1)
        self.assertEqual(result["detections"][0]["bbox"], [10, 10, 50, 50])

    def test_detect_people_includes_track_id(self):
        entries = [(0, 0.9, [10, 10, 50, 50])]
        result = self.understanding.detect_people(results=fake_results(entries, track_ids=[7]))
        self.assertEqual(result["detections"][0]["track_id"], 7)

    def test_detect_people_track_id_none_when_untracked(self):
        entries = [(0, 0.9, [10, 10, 50, 50])]
        result = self.understanding.detect_people(results=fake_results(entries))
        self.assertIsNone(result["detections"][0]["track_id"])

    def test_classify_scene_indoor(self):
        scene_type = self.understanding.classify_scene(results=fake_results([(1, 0.9, [0, 0, 20, 20])]))
        self.assertEqual(scene_type, "indoor")

    def test_classify_scene_urban(self):
        scene_type = self.understanding.classify_scene(results=fake_results([(2, 0.9, [0, 0, 20, 20])]))
        self.assertEqual(scene_type, "urban")

    def test_classify_scene_unknown_when_no_matching_labels(self):
        scene_type = self.understanding.classify_scene(results=fake_results([(0, 0.9, [0, 0, 20, 20])]))
        self.assertEqual(scene_type, "unknown")

    def test_infer_goals_falls_back_to_image_center_when_no_goal_objects(self):
        goals = self.understanding.infer_goals(
            image_width=640, image_height=480, results=fake_results([(0, 0.9, [0, 0, 20, 20])])
        )
        self.assertEqual(goals, [(320, 240)])

    def test_infer_goals_uses_goal_object_centers(self):
        goals = self.understanding.infer_goals(
            image_width=640, image_height=480, results=fake_results([(4, 0.9, [0, 0, 20, 20])])
        )
        self.assertEqual(goals, [(10, 10)])

    def test_detect_obstacles_returns_curated_labels_only(self):
        entries = [
            (0, 0.9, [10, 10, 50, 50]),        # person - not an obstacle
            (1, 0.9, [100, 100, 140, 160]),    # couch - obstacle
            (4, 0.9, [200, 200, 240, 240]),    # chair - goal label, not an obstacle
        ]
        result = self.understanding.detect_obstacles(results=fake_results(entries))

        self.assertEqual(result["num_obstacles"], 1)
        self.assertEqual(result["detections"][0]["label"], "couch")

    def test_detect_obstacles_uses_bbox_bottom_center_as_footprint(self):
        entries = [(1, 0.9, [100, 100, 140, 160])]  # couch: x center 120, bottom edge y=160
        result = self.understanding.detect_obstacles(results=fake_results(entries))

        self.assertEqual(result["detections"][0]["pixel_footprint"], (120, 160))

    def test_process_image_combines_people_scene_goals_and_obstacles(self):
        entries = [
            (0, 0.9, [10, 10, 50, 50]),       # person
            (1, 0.9, [60, 60, 90, 90]),       # couch -> indoor scene + obstacle
            (4, 0.9, [100, 100, 140, 140]),   # chair -> goal
        ]
        understanding = VisualSceneUnderstanding(detection_model=FakeDetectionModel(entries), camera=None)
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = understanding.process_image(image=image)

        self.assertEqual(result["num_people"], 1)
        self.assertEqual(result["scene_type"], "indoor")
        self.assertEqual(result["goals"], [(120, 120)])
        self.assertEqual(len(result["obstacles"]), 1)
        self.assertEqual(result["obstacles"][0]["label"], "couch")


if __name__ == '__main__':
    unittest.main()

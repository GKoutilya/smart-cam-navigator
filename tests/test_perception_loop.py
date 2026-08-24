import threading
import time
import unittest

import numpy as np

from main_pipeline import perception_loop
from src.planning.camera_path_planning import CameraPathPlanner
from src.planning.path_optimizer import PathOptimizer
from src.perception.target_tracker import TargetTracker
from src.utils.shared_state import SharedPerceptionState


class FakeCamera:
    def capture(self):
        return np.zeros((10, 10, 3), dtype=np.uint8)


class FakeSceneUnderstanding:
    """Stands in for VisualSceneUnderstanding: instant, synthetic per-frame results."""

    def process_image(self, image=None):
        return {
            "image": image,
            "people": [{"bbox": [0, 0, 10, 10], "confidence": 0.9, "track_id": 1}],
            "num_people": 1,
            "scene_type": "indoor",
            "goals": [(20, 20)],
            "obstacles": [],
        }


class IdentityMapper:
    """Stands in for GroundPlaneMapper with a trivial pixel<->world mapping."""

    def pixel_to_world(self, x, y):
        return (float(x), float(y))

    def world_to_pixel(self, x, y):
        return (float(x), float(y))


class TestPerceptionLoopThreading(unittest.TestCase):
    def test_publishes_updates_and_stops_promptly(self):
        shared_state = SharedPerceptionState()
        stop_event = threading.Event()

        thread = threading.Thread(
            target=perception_loop,
            args=(FakeCamera(), FakeSceneUnderstanding(), IdentityMapper(), TargetTracker(),
                  CameraPathPlanner(), PathOptimizer(), shared_state, stop_event),
            daemon=True,
        )
        thread.start()

        deadline = time.time() + 2.0
        while time.time() < deadline and not shared_state.snapshot().current_path:
            time.sleep(0.01)

        snapshot = shared_state.snapshot()
        self.assertIsNotNone(snapshot.image)
        self.assertEqual(snapshot.scene_type, "indoor")
        self.assertGreater(len(snapshot.current_path), 0)

        stop_event.set()
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())


if __name__ == '__main__':
    unittest.main()

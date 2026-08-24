import unittest

from src.utils.shared_state import SharedPerceptionState


class TestSharedPerceptionState(unittest.TestCase):
    def test_snapshot_reflects_last_update(self):
        state = SharedPerceptionState()
        state.update(scene_type="indoor", pixel_pose={"x": 1, "y": 2, "theta": 0.0})

        snapshot = state.snapshot()

        self.assertEqual(snapshot.scene_type, "indoor")
        self.assertEqual(snapshot.pixel_pose, {"x": 1, "y": 2, "theta": 0.0})

    def test_defaults_before_any_update(self):
        snapshot = SharedPerceptionState().snapshot()

        self.assertIsNone(snapshot.image)
        self.assertEqual(snapshot.people, [])
        self.assertEqual(snapshot.current_path, [])
        self.assertIsNone(snapshot.path_start_time)
        self.assertEqual(snapshot.perception_hz, 0.0)

    def test_later_update_does_not_mutate_earlier_snapshot(self):
        state = SharedPerceptionState()
        state.update(current_path=[(0.0, 0.0, 0.0)])

        first_snapshot = state.snapshot()
        state.update(current_path=[(0.0, 1.0, 1.0), (1.0, 2.0, 2.0)])
        second_snapshot = state.snapshot()

        self.assertEqual(first_snapshot.current_path, [(0.0, 0.0, 0.0)])
        self.assertEqual(second_snapshot.current_path, [(0.0, 1.0, 1.0), (1.0, 2.0, 2.0)])

    def test_partial_update_leaves_other_fields_untouched(self):
        state = SharedPerceptionState()
        state.update(scene_type="urban", perception_hz=5.0)
        state.update(scene_type="indoor")

        snapshot = state.snapshot()

        self.assertEqual(snapshot.scene_type, "indoor")
        self.assertEqual(snapshot.perception_hz, 5.0)


if __name__ == '__main__':
    unittest.main()

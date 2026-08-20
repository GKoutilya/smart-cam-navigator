import unittest

from src.perception.target_tracker import TargetTracker


def person(track_id, bbox):
    return {"bbox": bbox, "confidence": 0.9, "track_id": track_id}


class TestTargetTracker(unittest.TestCase):
    def test_acquires_first_target_when_none_held(self):
        tracker = TargetTracker()
        people = [person(1, [0, 0, 10, 10]), person(2, [100, 100, 110, 110])]

        target = tracker.update(people)

        self.assertEqual(target["track_id"], 1)
        self.assertEqual(tracker.target_track_id, 1)

    def test_continues_following_same_track_id(self):
        tracker = TargetTracker()
        tracker.update([person(1, [0, 0, 10, 10])])

        people = [person(2, [50, 50, 60, 60]), person(1, [5, 5, 15, 15])]
        target = tracker.update(people)

        self.assertEqual(target["track_id"], 1)
        self.assertEqual(target["bbox"], [5, 5, 15, 15])

    def test_returns_none_during_grace_period_when_target_missing(self):
        tracker = TargetTracker(max_missed_frames=3)
        tracker.update([person(1, [0, 0, 10, 10])])

        target = tracker.update([person(2, [50, 50, 60, 60])])

        self.assertIsNone(target)
        self.assertEqual(tracker.target_track_id, 1)

    def test_reacquires_nearest_target_after_grace_period_expires(self):
        tracker = TargetTracker(max_missed_frames=2)
        tracker.update([person(1, [0, 0, 10, 10])])  # last known center: (5, 5)

        other_people = [person(2, [50, 50, 60, 60])]
        for _ in range(3):
            target = tracker.update(other_people)

        self.assertEqual(target["track_id"], 2)
        self.assertEqual(tracker.target_track_id, 2)

    def test_no_target_when_nobody_detected(self):
        tracker = TargetTracker()
        target = tracker.update([])
        self.assertIsNone(target)
        self.assertIsNone(tracker.target_track_id)


if __name__ == '__main__':
    unittest.main()

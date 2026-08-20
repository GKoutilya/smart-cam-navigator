from typing import Any, Dict, List, Optional

from src.utils.helpers import calculate_distance


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


class TargetTracker:
    """Holds a single tracked person's identity across frames using YOLO track IDs.

    Reassigns only after the held track ID has been missing for more than
    `max_missed_frames` frames, and then prefers whichever detection is
    closest to the last known position - this avoids jumping between
    different people on a single frame's tracking hiccup or occlusion.
    """

    def __init__(self, max_missed_frames: int = 10):
        self.max_missed_frames = max_missed_frames
        self.target_track_id: Optional[int] = None
        self.last_known_pixel_position: Optional[tuple] = None
        self.missed_frames = 0

    def update(self, people: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if self.target_track_id is not None:
            for person in people:
                if person.get("track_id") == self.target_track_id:
                    self.missed_frames = 0
                    self.last_known_pixel_position = _bbox_center(person["bbox"])
                    return person
            self.missed_frames += 1

        if not people:
            return None

        if self.target_track_id is None or self.missed_frames > self.max_missed_frames:
            target = self._select_new_target(people)
            self.target_track_id = target.get("track_id")
            self.missed_frames = 0
            self.last_known_pixel_position = _bbox_center(target["bbox"])
            return target

        return None

    def _select_new_target(self, people: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.last_known_pixel_position is None:
            return people[0]
        return min(
            people,
            key=lambda person: calculate_distance(_bbox_center(person["bbox"]), self.last_known_pixel_position),
        )

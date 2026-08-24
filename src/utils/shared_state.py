import threading
from types import SimpleNamespace


class SharedPerceptionState:
    """Lock-protected handoff from the perception/planning thread to the
    control/display loop. The perception thread always publishes freshly
    constructed lists (never mutates one already handed out), so snapshot()
    needs no deep copy to stay safe against later updates.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.image = None
        self.people = []
        self.scene_type = "unknown"
        self.pixel_pose = None
        self.pixel_goals = []
        self.current_path = []  # [(t, x, y), ...]
        self.path_start_time = None
        self.perception_hz = 0.0
        self.world_obstacles = []  # [(x, y), ...]

    def update(self, **fields) -> None:
        with self._lock:
            for key, value in fields.items():
                setattr(self, key, value)

    def snapshot(self) -> SimpleNamespace:
        with self._lock:
            return SimpleNamespace(
                image=self.image,
                people=self.people,
                scene_type=self.scene_type,
                pixel_pose=self.pixel_pose,
                pixel_goals=self.pixel_goals,
                current_path=self.current_path,
                path_start_time=self.path_start_time,
                perception_hz=self.perception_hz,
                world_obstacles=self.world_obstacles,
            )

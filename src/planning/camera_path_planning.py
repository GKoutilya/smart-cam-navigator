from src.planning.costmap import Costmap
from src.planning.grid_astar import find_path


class CameraPathPlanner:
    def __init__(self, costmap: Costmap = None):
        self.costmap = costmap or Costmap()

    def set_costmap(self, costmap: Costmap) -> None:
        self.costmap = costmap

    def plan_path(self, start, goals):
        """
            Plans a path from start through multiple goals, routing around
            obstacles in the current costmap.
        """
        path = [start]
        for goal in goals:
            segment = self.plan_segment(path[-1], goal)
            path.extend(segment[1:])

        return path

    def plan_segment(self, start, goal):
        return find_path(start, goal, self.costmap)

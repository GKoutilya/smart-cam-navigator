import numpy as np
from src.planning.path_optimizer import PathOptimizer

class CameraPathPlanner:
    def __init__(self):
        self.optimizer = PathOptimizer()

    def plan_path(self, start, goals, num_points=20):
        """
            Plans a path from start through multiple goals.
        """
        path = [start]
        for goal in goals:
            segment = self.plan_segment(path[-1], goal, num_points)
            path.extend(segment[1:])

        return path
    
    def plan_segment(self, start, goal, num_points):
        x_vals = np.linspace(start[0], goal[0], num_points)
        y_vals = np.linspace(start[1], goal[1], num_points)
        return list(zip(x_vals, y_vals))

    def optimize_path(self, path):
        """
            Optimize the path using the PathOptimizer module.
        """
        return self.optimizer.optimize_path(path)
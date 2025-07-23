import numpy as np
from src.planning.path_optimizer import PathOptimizer

class CameraPathPlanner:
    def __init__(self, image_width, image_height):
        self.image_width = image_width
        self.image_height = image_height
        self.optimizer = PathOptimizer()

    def extract_targets(self, person_boxes):
        """
            Extracts central points (targets) from person bounding boxes.
            Args:
                person_boxes: list of [x1, y1, x2, y2] bounding boxes.
            Returns:
                list of (x, y) positions normalized between 0-1.
        """
        targets = []

        for box in person_boxes:
            x_center = (((box[0] + box[2])) / 2) / self.image_width
            y_center = (((box[1] + box[3])) / 2) / self.image_height
            targets.append((x_center, y_center))

        return targets

    def plan_path(self, start, goal, num_points=20):
        """
            Plans a linear path from start to goal with intermediate points.
        """
        x_vals = np.linspace(start[0], goal[0], num_points)
        y_vals = np.linspace(start[1], goal[1], num_points)
        path = list(zip(x_vals, y_vals))

        return path

    def optimize_path(self, path):
        """
            Optimize the path using the PathOptimizer module.
        """
        return self.optimizer.optimize_path(path)
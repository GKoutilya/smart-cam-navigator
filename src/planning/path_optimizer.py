import math

class PathOptimizer:
    def __init__(self):
        pass

    def smooth_path(self, path, weight=0.5):
        """
            Smooths the path using averaging technique.
        """
        if len(path) < 3:
            return path

        smoothed_path = [path[0]]
        for i in range(1, len(path) - 1):
            prev_point = path[i-1]
            curr_point = path[i]
            next_point = path[i+1]

            smoothed_x = (
                weight * prev_point[0] +
                (1 - 2 * weight) * curr_point[0] +
                weight * next_point[0]
            )

            smoothed_y = (
                weight * prev_point[1] +
                (1 - 2 * weight) * curr_point[1] +
                weight * next_point[1]
            )

            smoothed_path.append((smoothed_x, smoothed_y))
        
        smoothed_path.append(path[-1])

        return smoothed_path

    def ensure_feasibility(self, path, max_step_size=5.0):
        """
            Ensures no two consecutive points are too far apart.
        """
        for i in range(len(path) - 1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            dist = math.hypot(dx, dy)
            if dist > max_step_size:
                return False

        return True
    
    def path_length(self, path):
        return sum(math.hypot(path[i+1][0], path[i+1][1]-path[i][1]) for i in range(len(path) - 1))

    def optimize_path(self, path, smoothing_method="average"):
        if not self.ensure_feasibility(path):
            raise ValueError("Path is not feasible for execution.")
        if smoothing_method == "average":
            return self.smooth_path(path)
        
        return path
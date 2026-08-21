def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def normalize_vector(vector):
    """Normalize a 2D vector."""
    length = calculate_distance((0, 0), vector)
    if length == 0:
        return (0, 0)
    return (vector[0] / length, vector[1] / length)

def interpolate_points(point1, point2, num_points):
    """Interpolate between two points."""
    return [
        (
            point1[0] + (point2[0] - point1[0]) * t / (num_points - 1),
            point1[1] + (point2[1] - point1[1]) * t / (num_points - 1)
        )
        for t in range(num_points)
    ]

def sample_timed_path(timed_path, t):
    """Linearly interpolates a [(t, x, y), ...] path at time t, clamped to the path's ends."""
    if not timed_path:
        raise ValueError("timed_path must not be empty.")

    if t <= timed_path[0][0]:
        return (timed_path[0][1], timed_path[0][2])
    if t >= timed_path[-1][0]:
        return (timed_path[-1][1], timed_path[-1][2])

    for (t0, x0, y0), (t1, x1, y1) in zip(timed_path, timed_path[1:]):
        if t0 <= t <= t1:
            if t1 == t0:
                return (x1, y1)
            ratio = (t - t0) / (t1 - t0)
            return (x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio)

    return (timed_path[-1][1], timed_path[-1][2])


def visualize_path_from_csv(csv_path):
    """Load path from CSV and visualize it using matplotlib."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import os

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV path {csv_path} not found.")

    df = pd.read_csv(csv_path)
    if 'x' not in df.columns or 'y' not in df.columns:
        raise ValueError("CSV must contain 'x' and 'y' columns.")

    path = list(zip(df['x'], df['y']))
    x, y = zip(*path)

    plt.plot(x, y, marker='o')
    plt.title('Robot Run Path from CSV')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()
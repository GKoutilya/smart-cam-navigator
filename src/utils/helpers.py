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

def visualize_path(path):
    """Visualize a path using a simple plotting method."""
    import matplotlib.pyplot as plt

    x, y = zip(*path)
    plt.plot(x, y, marker='o')
    plt.title('Camera Path')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    plt.show()
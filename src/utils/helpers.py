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

def visualize_path_from_csv(csv_path, save_plot=True):
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

    '''if save_plot:
        plot_name = os.path.splitext(csv_path)[0] + "_path_plot.png"
        plt.savefig(plot_name)
        print(f"Plot saved to: {plot_name}")
    else:
        plt.show()'''
    
def draw_annotations(image, people, pose, scene_type):
    import cv2

    for p in people['detections']:
        x1, y1, x2, y2 = p['bbox']
        conf = p['confidence']
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f"{conf}", (x1, y1 -10),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        
    cv2.putText(image, f"Scene: {scene_type}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    pose_text = f"Pose: x={pose['x']} y={pose['y']} theta={pose['theta']:.2f}"
    cv2.putText(image, pose_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return image
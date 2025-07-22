from typing import Tuple, List, Dict, Any

# Define types for scene representations
SceneType = Dict[str, Any]
CameraParameters = Dict[str, float]

# Define a type for a path represented as a list of coordinates
Path = List[Tuple[float, float]]

# Define a type for a detection result
DetectionResult = Dict[str, Any]  # This can include bounding boxes, class labels, etc.

# Exporting the types for use in other modules
__all__ = ['SceneType', 'CameraParameters', 'Path', 'DetectionResult']
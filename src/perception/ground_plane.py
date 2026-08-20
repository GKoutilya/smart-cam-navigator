from typing import List, Tuple
import numpy as np
import cv2


def compute_homography(image_points: List[Tuple[float, float]],
                        world_points: List[Tuple[float, float]]) -> np.ndarray:
    """Computes the pixel-to-world homography from 4 point correspondences."""
    if len(image_points) != 4 or len(world_points) != 4:
        raise ValueError("Exactly 4 point correspondences are required.")

    src = np.array(image_points, dtype=np.float32)
    dst = np.array(world_points, dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _apply(matrix: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    point = matrix @ np.array([x, y, 1.0])
    w = point[2]
    return (point[0] / w, point[1] / w)


class GroundPlaneMapper:
    """Maps between camera pixel coordinates and a calibrated ground-plane frame."""

    def __init__(self, homography: np.ndarray):
        self.homography = homography
        self.inverse_homography = np.linalg.inv(homography)

    @classmethod
    def from_file(cls, path: str) -> "GroundPlaneMapper":
        homography = np.load(path)
        return cls(homography)

    def save(self, path: str) -> None:
        np.save(path, self.homography)

    def pixel_to_world(self, px: float, py: float) -> Tuple[float, float]:
        return _apply(self.homography, px, py)

    def world_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        return _apply(self.inverse_homography, x, y)

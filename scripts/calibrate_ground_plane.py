"""
One-time ground-plane calibration for a static webcam.

Mark a real rectangle on the floor (tape, a rug, floor tiles - anything with
straight, right-angled edges). Run this script, click its 4 corners in the
live feed in order (top-left, top-right, bottom-right, bottom-left), then
enter the rectangle's width/height in whatever units you want to plan in
(defaults to a 1.0 x 1.0 unit square - no tape measure required).

Usage:
    python scripts/calibrate_ground_plane.py [--width W] [--height H] [--cam-index N]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.perception.camera import WebcamCamera
from src.perception.ground_plane import GroundPlaneMapper, compute_homography

WINDOW_NAME = "Ground-plane calibration - click 4 floor corners (TL, TR, BR, BL)"
DEFAULT_OUTPUT_PATH = os.path.join("calibration", "homography.npy")


def _collect_clicks(camera: WebcamCamera) -> list:
    clicked_points = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
            clicked_points.append((float(x), float(y)))

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while len(clicked_points) < 4:
        frame = camera.capture()
        if frame is None:
            continue

        display = frame.copy()
        for point in clicked_points:
            cv2.circle(display, (int(point[0]), int(point[1])), 6, (0, 255, 0), -1)
        cv2.putText(display, f"Clicked {len(clicked_points)}/4 corners", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow(WINDOW_NAME, display)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            raise KeyboardInterrupt("Calibration cancelled.")

    cv2.destroyAllWindows()
    return clicked_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=float, default=1.0,
                         help="Declared width of the marked rectangle (default: 1.0)")
    parser.add_argument("--height", type=float, default=1.0,
                         help="Declared height of the marked rectangle (default: 1.0)")
    parser.add_argument("--cam-index", type=int, default=0)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    camera = WebcamCamera(cam_index=args.cam_index)
    try:
        print("Click the rectangle's corners in order: top-left, top-right, bottom-right, bottom-left.")
        image_points = _collect_clicks(camera)
    finally:
        camera.release()

    world_points = [
        (0.0, 0.0),
        (args.width, 0.0),
        (args.width, args.height),
        (0.0, args.height),
    ]

    homography = compute_homography(image_points, world_points)
    mapper = GroundPlaneMapper(homography)

    reprojection_error = 0.0
    for image_point, expected in zip(image_points, world_points):
        actual = mapper.pixel_to_world(*image_point)
        reprojection_error = max(reprojection_error, np.hypot(actual[0] - expected[0], actual[1] - expected[1]))

    print(f"Reprojection error (max corner deviation): {reprojection_error:.4f} world units")
    if reprojection_error > 0.05 * max(args.width, args.height):
        print("Warning: reprojection error is large relative to the rectangle size - "
              "consider re-running calibration with more careful clicks.")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mapper.save(args.output)
    print(f"Saved calibration to {args.output}")


if __name__ == "__main__":
    main()

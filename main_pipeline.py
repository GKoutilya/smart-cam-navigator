from src.perception.visual_scene_understanding import VisualSceneUnderstanding
from src.planning.camera_path_planning import CameraPathPlanner
from src.planning.path_optimizer import PathOptimizer
from src.action.robot_controller import RobotController
from src.utils.helpers import visualize_path_from_csv
from src.perception.camera import WebcamCamera
from src.perception.ground_plane import GroundPlaneMapper
from src.perception.target_tracker import TargetTracker
from ultralytics import YOLO
import numpy as np
import time
import cv2
import os

CALIBRATION_PATH = os.path.join("calibration", "homography.npy")

def pixel_pose_from_person(person):
    x1, y1, x2, y2 = person['bbox']
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return {"x": center_x, "y": center_y, "theta": 0.0}

def main():
    if not os.path.exists(CALIBRATION_PATH):
        raise FileNotFoundError(
            f"No ground-plane calibration found at '{CALIBRATION_PATH}'. "
            "Run `python scripts/calibrate_ground_plane.py` first."
        )
    mapper = GroundPlaneMapper.from_file(CALIBRATION_PATH)

    camera = WebcamCamera(cam_index=0)
    detection_model = YOLO("yolov8n.pt")
    scene_understanding = VisualSceneUnderstanding(detection_model=detection_model, camera=camera)

    image = camera.capture()
    if image is None:
        raise ValueError("Failed to load image from camera.")

    path_planner = CameraPathPlanner()
    path_optimizer = PathOptimizer()
    controller = RobotController(vision=scene_understanding)
    target_tracker = TargetTracker()

    execution_index = 0
    current_path = []  # world-space (ground-plane) coordinates
    REPLAN_INTERVAL_SEC = 1.5
    last_replan_time = time.time()
    pixel_goals = []  # kept in pixel space, only used for drawing
    pixel_pose = None  # last known pixel-space pose of the tracked target

    prev_time = time.time()

    while True:
        image = camera.capture()
        if image is None:
            continue

        # Single capture, single detection pass per frame: people, scene type,
        # and goals are all derived from the same YOLO result on the same frame.
        semantic_info = scene_understanding.process_image(image=image)
        people = semantic_info["people"]
        scene_type = semantic_info["scene_type"]

        # Holds onto one tracked person across frames instead of re-picking
        # people[0]; returns None both when nobody is around and during the
        # tracker's grace period for a momentarily-missing target, so the
        # loop below must keep running (path execution, display, quit-check)
        # rather than skipping the frame entirely.
        target = target_tracker.update(people)
        current_time = time.time()

        if target is not None:
            pixel_pose = pixel_pose_from_person(target)
            world_pose = mapper.pixel_to_world(pixel_pose['x'], pixel_pose['y'])
            controller.current_position = world_pose

            # Replan only at interval or if no path
            if current_time - last_replan_time >= REPLAN_INTERVAL_SEC or not current_path:
                pixel_goals = semantic_info["goals"]
                if pixel_goals:
                    world_goals = [mapper.pixel_to_world(gx, gy) for gx, gy in pixel_goals]
                    raw_path = path_planner.plan_path(start=controller.current_position, goals=world_goals)
                    raw_path = [(float(x), float(y)) for x, y in raw_path]
                    current_path = path_optimizer.optimize_path(raw_path)
                    execution_index = 0  # Reset to start new path
                last_replan_time = current_time

        # Execute one step of the path per frame
        if current_path and execution_index < len(current_path):
            next_pos = current_path[execution_index]
            controller.move_to(next_pos)
            controller.log_movement(next_pos)
            controller.current_position = next_pos  # update after moving
            execution_index += 1

        # Visualization: draw current_path, pose, goals, etc.
        annotated = image.copy()
        for person in people:
            x1, y1, x2, y2 = person['bbox']
            conf = person.get('confidence', 1.0)
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated, f"person {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if pixel_pose is not None:
            cv2.circle(annotated, (pixel_pose['x'], pixel_pose['y']), 10, (0, 255, 0), -1)

        for goal in pixel_goals:
            cv2.circle(annotated, (int(goal[0]), int(goal[1])), 8, (255, 0, 0), -1)

        if current_path and len(current_path) > execution_index:
            remaining_pts = current_path[execution_index:]
            pixel_pts = [mapper.world_to_pixel(x, y) for x, y in remaining_pts]
            pts = [(int(x), int(y)) for x, y in pixel_pts]
            if len(pts) > 1:
                cv2.polylines(annotated, [np.array(pts, dtype=np.int32)], False, (0, 0, 255), 2)

        fps = 1 / (current_time - prev_time) if current_time != prev_time else 0.0
        prev_time = current_time
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated, f"Scene: {scene_type}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Robot Perception + Planning", annotated)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()
    visualize_path_from_csv(csv_path=r"logs\run.csv")

if __name__ == "__main__":
    main()

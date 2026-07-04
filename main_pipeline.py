from src.perception.visual_scene_understanding import VisualSceneUnderstanding
from src.planning.camera_path_planning import CameraPathPlanner
from src.planning.path_optimizer import PathOptimizer
from src.action.robot_controller import RobotController
from src.utils.helpers import visualize_path_from_csv
from src.perception.camera import WebcamCamera
from ultralytics import YOLO
import numpy as np
import time
import cv2

def get_robot_pose_from_person(people):
    if not people:
        return None
    person = people[0]
    x1, y1, x2, y2 = person['bbox']
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return {"x": center_x, "y": center_y, "theta": 0.0}

def main():
    camera = WebcamCamera(cam_index=0)
    detection_model = YOLO("yolov8n.pt")
    scene_understanding = VisualSceneUnderstanding(detection_model=detection_model, camera=camera)

    image = camera.capture()
    if image is None:
        raise ValueError("Failed to load image from camera.")

    height, width = image.shape[:2]
    path_planner = CameraPathPlanner(width, height)
    path_optimizer = PathOptimizer()
    controller = RobotController(vision=scene_understanding)

    execution_index = 0
    current_path = []
    REPLAN_INTERVAL_SEC = 1.5
    last_replan_time = time.time()
    goals = []

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

        pose = get_robot_pose_from_person(people)
        if pose is None:
            print("No person detected; skipping frame.")
            continue

        controller.current_position = (pose['x'], pose['y'])
        current_time = time.time()

        # Replan only at interval or if no path
        if current_time - last_replan_time >= REPLAN_INTERVAL_SEC or not current_path:
            goals = semantic_info["goals"]
            if goals:
                raw_path = path_planner.plan_path(start=controller.current_position, goals=goals)
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

        cv2.circle(annotated, (pose['x'], pose['y']), 10, (0, 255, 0), -1)

        for goal in goals:
            cv2.circle(annotated, (int(goal[0]), int(goal[1])), 8, (255, 0, 0), -1)

        if current_path and len(current_path) > execution_index:
            remaining_pts = current_path[execution_index:]
            pts = [(int(x), int(y)) for x, y in remaining_pts]
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

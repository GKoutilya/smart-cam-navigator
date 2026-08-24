from src.perception.visual_scene_understanding import VisualSceneUnderstanding
from src.planning.camera_path_planning import CameraPathPlanner
from src.planning.path_optimizer import PathOptimizer
from src.action.robot_controller import RobotController
from src.utils.helpers import visualize_path_from_csv, sample_timed_path
from src.utils.shared_state import SharedPerceptionState
from src.perception.camera import WebcamCamera
from src.perception.ground_plane import GroundPlaneMapper
from src.perception.target_tracker import TargetTracker
from src.planning.costmap import Costmap
from src.action.simulated_robot import SimulatedRobot
from ultralytics import YOLO
import numpy as np
import threading
import time
import cv2
import os

CALIBRATION_PATH = os.path.join("calibration", "homography.npy")
REPLAN_INTERVAL_SEC = 1.5
CONTROL_RATE_HZ = 30

def pixel_pose_from_person(person):
    x1, y1, x2, y2 = person['bbox']
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return {"x": center_x, "y": center_y, "theta": 0.0}


def perception_loop(camera, scene_understanding, mapper, target_tracker,
                     path_planner, path_optimizer, shared_state, stop_event,
                     replan_interval_sec=REPLAN_INTERVAL_SEC):
    """Runs capture -> detection -> tracking -> (re)planning at whatever rate
    the camera/model allow, publishing results for the control loop to
    consume independently. Owns all perception/planning state; never touches
    RobotController - the control loop is the only thing that drives it.
    """
    current_path = []
    path_start_time = None
    last_replan_time = time.time()
    pixel_pose = None
    pixel_goals = []

    tick_count = 0
    rate_window_start = time.time()
    perception_hz = 0.0

    while not stop_event.is_set():
        image = camera.capture()
        if image is None:
            continue

        # Single capture, single detection pass per frame: people, scene type,
        # goals, and obstacles are all derived from the same YOLO result.
        semantic_info = scene_understanding.process_image(image=image)
        people = semantic_info["people"]
        scene_type = semantic_info["scene_type"]

        # Holds onto one tracked person across frames instead of re-picking
        # people[0]; returns None both when nobody is around and during the
        # tracker's grace period for a momentarily-missing target.
        target = target_tracker.update(people)
        current_time = time.time()

        # Published every frame regardless of replanning - a newly-visible
        # obstacle should update physical collision immediately, the same way
        # it's already visible on screen immediately.
        world_obstacles = [mapper.pixel_to_world(*obstacle["pixel_footprint"]) for obstacle in semantic_info["obstacles"]]

        if target is not None:
            pixel_pose = pixel_pose_from_person(target)
            world_pose = mapper.pixel_to_world(pixel_pose['x'], pixel_pose['y'])

            # Replan only at interval or if no path
            if current_time - last_replan_time >= replan_interval_sec or not current_path:
                pixel_goals = semantic_info["goals"]
                if pixel_goals:
                    world_goals = [mapper.pixel_to_world(gx, gy) for gx, gy in pixel_goals]

                    costmap = Costmap()
                    for ox, oy in world_obstacles:
                        costmap.add_obstacle(ox, oy)
                    path_planner.set_costmap(costmap)

                    raw_path = path_planner.plan_path(start=world_pose, goals=world_goals)
                    raw_path = [(float(x), float(y)) for x, y in raw_path]
                    current_path = path_optimizer.optimize_path(raw_path)
                    path_start_time = current_time
                last_replan_time = current_time

        tick_count += 1
        if current_time - rate_window_start >= 1.0:
            perception_hz = tick_count / (current_time - rate_window_start)
            tick_count = 0
            rate_window_start = current_time

        shared_state.update(
            image=image,
            people=people,
            scene_type=scene_type,
            pixel_pose=pixel_pose,
            pixel_goals=pixel_goals,
            current_path=current_path,
            path_start_time=path_start_time,
            perception_hz=perception_hz,
            world_obstacles=world_obstacles,
        )


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
    simulated_robot = SimulatedRobot()
    controller = RobotController(vision=scene_understanding, robot=simulated_robot)
    target_tracker = TargetTracker()
    shared_state = SharedPerceptionState()
    stop_event = threading.Event()

    perception_thread = threading.Thread(
        target=perception_loop,
        args=(camera, scene_understanding, mapper, target_tracker,
              path_planner, path_optimizer, shared_state, stop_event),
        daemon=True,
    )
    perception_thread.start()

    control_dt = 1.0 / CONTROL_RATE_HZ
    prev_time = time.time()

    try:
        while True:
            tick_start = time.time()
            snapshot = shared_state.snapshot()

            if snapshot.image is None:
                time.sleep(control_dt)
                continue

            current_time = time.time()

            # Follow the time-parameterized trajectory by elapsed real time,
            # not by however often a new camera frame happens to arrive - this
            # is what lets control run at CONTROL_RATE_HZ regardless of how
            # slow perception is.
            elapsed = current_time - snapshot.path_start_time if snapshot.current_path else None
            if snapshot.current_path:
                simulated_robot.set_obstacles(snapshot.world_obstacles)
                next_pos = sample_timed_path(snapshot.current_path, elapsed)
                controller.move_to(next_pos, dt=control_dt)
                # Logs the robot's actual simulated position, not the raw
                # commanded point - the two can now legitimately differ since
                # the robot has real inertia and a turning radius.
                controller.log_movement(controller.current_position)

            # Visualization: draw current_path, pose, goals, etc.
            annotated = snapshot.image.copy()
            for person in snapshot.people:
                x1, y1, x2, y2 = person['bbox']
                conf = person.get('confidence', 1.0)
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated, f"person {conf:.2f}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if snapshot.pixel_pose is not None:
                cv2.circle(annotated, (snapshot.pixel_pose['x'], snapshot.pixel_pose['y']), 10, (0, 255, 0), -1)

            for goal in snapshot.pixel_goals:
                cv2.circle(annotated, (int(goal[0]), int(goal[1])), 8, (255, 0, 0), -1)

            if snapshot.current_path:
                remaining_pts = [(x, y) for t, x, y in snapshot.current_path if t >= elapsed]
                pixel_pts = [mapper.world_to_pixel(x, y) for x, y in remaining_pts]
                pts = [(int(x), int(y)) for x, y in pixel_pts]
                if len(pts) > 1:
                    cv2.polylines(annotated, [np.array(pts, dtype=np.int32)], False, (0, 0, 255), 2)

                # The simulated robot's actual position - can trail behind or
                # turn toward the commanded point instead of snapping to it.
                actual_px, actual_py = mapper.world_to_pixel(*controller.current_position)
                cv2.circle(annotated, (int(actual_px), int(actual_py)), 6, (0, 255, 255), -1)

            control_hz = 1 / (current_time - prev_time) if current_time != prev_time else 0.0
            prev_time = current_time
            cv2.putText(annotated, f"Control Hz: {control_hz:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated, f"Perception Hz: {snapshot.perception_hz:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(annotated, f"Scene: {snapshot.scene_type}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow("Robot Perception + Planning", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            tick_elapsed = time.time() - tick_start
            time.sleep(max(0.0, control_dt - tick_elapsed))
    finally:
        stop_event.set()
        perception_thread.join(timeout=2.0)
        camera.release()
        cv2.destroyAllWindows()

    visualize_path_from_csv(csv_path=r"logs\run.csv")

if __name__ == "__main__":
    main()

from typing import List, Tuple
from src.perception.visual_scene_understanding import VisualSceneUnderstanding
from src.planning.camera_path_planning import CameraPathPlanner
from typing import Callable, Optional
from datetime import datetime
import time
import csv
import os

class RobotController:
    def __init__(self, robot=None, vision=None, log_dir="logs", log_name="run.csv",
                 on_step: Optional[Callable]=None):
        self.robot = robot
        self.vision = vision or VisualSceneUnderstanding(detection_model=None)
        self.current_step = 0
        self.current_position = (0,0)
        self.image_captured = []

        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.abspath(os.path.join(log_dir, log_name))

        if not os.path.exists(self.log_path) or os.path.getsize(self.log_path) == 0:
            with open(self.log_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "x", "y"])
        
        self.on_step = on_step

    def move_to(self, position: Tuple[float, float]):
        # Code to move the robot to the specified position
        print(f"Moving to position {position}")
        time.sleep(0.1)
        self.current_position = position

    def log_movement(self, position):
        print(f"Logging position: {position} to {self.log_path}")
        with open(self.log_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), position[0], position[1]])
            f.flush()

    def execute_path(self, path: List[Tuple[float, float]]):
        self.image_captured.clear()
        for position in path:
            try:
                self.move_to(position)
                self.log_movement(position)
                image = self.vision.capture_image()
                scene_info = self.vision.process_image()
                self.image_captured.append({
                    "position": position,
                    "image": image,
                    "scene_info": scene_info
                })
                if self.on_step:
                    self.on_step(position, image, scene_info)
            except Exception as e:
                print(f'Error at position {position}: {e}')
                self.stop()
                break
        return True
    
    def plan_and_act(self, planner: CameraPathPlanner, goals: List[Tuple[float, float]], num_points=20):
        """
            Plans a path using the given planner and executed it.

            Args:
                planner (CameraPathPlanner): An instance with a 'plan_path(start, goals)' method.
                goals (List[Tuple[float, float]]): List of goal coordinates to reach.
                num_points(int): Number of points for each path segment
        """
        start = self.current_position
        if not goals:
            print("No goals provided.")
            return False
        
        print(f"Planning path from {start} to {goals}")
        path = planner.plan_path(start=start, goals=goals, num_points=num_points)

        if not path or len(path) < 2:
            print("Failed to generate a valid path.")
            return False
        
        print(f"Executing path with {len(path)} steps...")
        success = self.execute_path(path)
        return success

    def stop(self):
        # Code to stop the robot's movement
        print("Robot stopped.")

    def get_status(self) -> str:
        # Code to return the current status of the robot
        return "Robot is operational."
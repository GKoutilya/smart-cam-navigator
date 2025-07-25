from typing import List, Tuple, Any
from src.perception.visual_scene_understanding import VisualSceneUnderstanding
from src.utils.helpers import visualize_path_from_csv
from typing import Callable, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import time
import cv2
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

    def capture_image(self) -> Any:
        return cv2.imread("sample_image.jpg")

    def execute_path(self, path: List[Tuple[float, float]]):
        self.image_captured.clear()
        for position in path:
            try:
                self.move_to(position)
                self.log_movement(position)
                image = self.vision.capture_image()
                scene_info = self.vision.process_image(image)
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

    def stop(self):
        # Code to stop the robot's movement
        print("Robot stopped.")

    def get_status(self) -> str:
        # Code to return the current status of the robot
        return "Robot is operational."
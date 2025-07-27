from src.perception.visual_scene_understanding import VisualSceneUnderstanding
from src.planning.camera_path_planning import CameraPathPlanner
from src.planning.path_optimizer import PathOptimizer
from src.action.robot_controller import RobotController
from src.utils.helpers import visualize_path_from_csv, draw_annotations
from src.perception.camera import HEICCamera
from ultralytics import YOLO
import cv2

def main():
    camera = HEICCamera(r"C:\Users\kouti\Downloads\IMG_4525.HEIC")
    image = cv2.cvtColor(camera.capture(), cv2.COLOR_BGR2RGB)

    if image is None:
        raise ValueError("Failed to load image from camera.")

    detection_model = YOLO("yolov8n.pt")
    scene_understanding = VisualSceneUnderstanding(detection_model=detection_model, camera=camera)

    height, width = image.shape[:2]
    path_planner = CameraPathPlanner(width, height)
    path_optimizer = PathOptimizer()
    controller = RobotController(vision=scene_understanding)

    goals = [(100, 200), (400, 300)] # Random at the moment
    semantic_info = scene_understanding.process_image()
    pose_dict = semantic_info['pose'] # Also random at the moment
    start = (pose_dict['x'], pose_dict['y'])
    raw_path = path_planner.plan_path(start, goals)
    raw_path = [(float(x), float(x)) for x, y in raw_path]
    print("Raw Path:", raw_path)
    print("Length of path:", len(raw_path))
    optimized_path = path_optimizer.optimize_path(raw_path)
    controller.execute_path(optimized_path)

    visualize_path_from_csv(csv_path=r"logs\run.csv")
    annotated = draw_annotations(image,
                                 semantic_info['people'],
                                 semantic_info['pose'],
                                 semantic_info['scene_type'])
    
    cv2.imshow("Scene", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
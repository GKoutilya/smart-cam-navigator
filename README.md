# 🔍 Visual Scene Understanding and Robot Path Planning Pipeline

This project simulates an end-to-end robotics perception and control pipeline. It processes synthetic camera inputs, detects obstacles, builds a scene representation, plans a path to a goal, and issues movement commands — all without real hardware or external models.

## 🚀 Features

- **Camera Simulation**: Simulated environment frames for perception testing.
- **Visual Scene Understanding**: Parses object detections to build semantic maps.
- **Path Planning**: Generates a path around obstacles using simple grid-based logic.
- **Robot Control**: Issues directional commands based on planned waypoints.
- **Modular Design**: Each stage is built independently for testing, debugging, and potential extension.

## 🧠 Pipeline Overview

```text
[Simulated Camera Frame]
          ↓
[Object Detection Model]
          ↓
[Visual Scene Understanding]
          ↓
[Path Planner]
          ↓
[Robot Controller]
````

> If you don’t have `camera_simulator.py` or `dummy_detection_model.py`, remove or replace their usage in `main_pipeline.py`.

## ⚙️ Getting Started

1. **Install requirements** (if any):

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the pipeline**:

   ```bash
   python main_pipeline.py
   ```

3. **Run unit tests**:

   ```bash
   python test_visual_scene_understanding.py
   ```

## 🧪 Current Limitations

* No real-time camera input (all synthetic or mocked).
* Object detection model is simulated — no YOLO, OpenCV, etc.
* Path planner uses a simple greedy grid-based algorithm.

## 🛠️ Next Steps

* Integrate a real object detection model (e.g., YOLOv5).
* Add real sensor inputs or simulation integration (e.g., Gazebo).
* Enhance the planner with A\*, RRT, or D\*.
* Add logging, metrics, or a visualization UI.

## 👤 Author

Koutilya Ganapathiraju
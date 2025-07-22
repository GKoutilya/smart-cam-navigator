# ml-perception-action-system

This project implements a full machine learning perception-to-action system for autonomous visual scene understanding and camera path planning in human-centric robotics. The system is designed to enable robots to understand their environment visually and plan optimal paths for camera movements.

## Project Structure

- **src/**: Contains the main source code for the project.
  - **perception/**: Implements visual scene understanding functionalities.
    - **visual_scene_understanding.py**: Main functionality for pose estimation and detection of people in images.
    - **models/**: Contains data structures and models for representing and classifying visual scenes.
      - **scene_model.py**: Defines classes for different scene types and methods for training and inference.
  - **planning/**: Implements camera path planning algorithms.
    - **camera_path_planning.py**: Contains the `CameraPathPlanner` class for planning and optimizing camera paths.
    - **path_optimizer.py**: Includes algorithms for optimizing planned camera paths.
  - **action/**: Responsible for executing planned camera paths.
    - **robot_controller.py**: Defines the `RobotController` class for controlling robotic camera movements.
  - **utils/**: Provides utility functions and helper methods.
    - **helpers.py**: Contains functions for data processing and visualization.
  - **types/**: Exports various types and interfaces used across the project.
    - **index.py**: Defines data structures for scene representations and camera parameters.

- **tests/**: Contains unit tests for the project modules.
  - **test_visual_scene_understanding.py**: Tests for the visual scene understanding module.
  - **test_camera_path_planning.py**: Tests for the camera path planning module.
  - **test_robot_controller.py**: Tests for the robot controller module.

- **requirements.txt**: Lists the dependencies required for the project.

- **README.md**: Documentation for the project.

- **setup.py**: Configuration file for packaging the project.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. Import the necessary modules from the `src` directory.
2. Use the `CameraPathPlanner` to plan and optimize camera paths based on visual scene understanding.
3. Control the robotic camera using the `RobotController` to execute the planned paths.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
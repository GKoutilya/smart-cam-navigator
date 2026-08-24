# 🧠 Real-Time Perception-to-Action Pipeline for Robotic Navigation

[![Tests](https://github.com/GKoutilya/smart-cam-navigator/actions/workflows/tests.yml/badge.svg)](https://github.com/GKoutilya/smart-cam-navigator/actions/workflows/tests.yml)

This project demonstrates a modular, real-time perception → planning → control pipeline designed for indoor robotic navigation using live webcam input. A single static webcam is calibrated to a real ground-plane frame, tracks one person persistently, plans an obstacle-aware path with a real velocity/acceleration-limited trajectory, and drives a physically simulated differential-drive robot toward it — all decoupled across perception and control rates, with visual feedback and logging.

## 🎯 Key Features

| Module | Description |
|--------|-------------|
| 🔍 **Perception** | YOLOv8 for object/person detection, scene classification, and curated-label obstacle detection |
| 📍 **Ground-Plane Pose** | One-time homography calibration converts pixel detections to real ground-plane coordinates, not raw image pixels |
| 🎯 **Persistent Tracking** | ByteTrack-based target tracking holds one person's identity across frames instead of re-picking whoever YOLO lists first |
| 🧠 **Goal Inference** | Automatically selects a semantic goal (e.g., door, chair) based on detections in the scene |
| 🗺️ **Obstacle-Aware Planning** | A\* search over a costmap built from detected furniture, plus a trapezoidal-velocity-profile trajectory optimizer (not simple point-to-point interpolation) |
| ⚙️ **Decoupled Rates** | Perception/planning runs on a background thread at whatever rate detection allows; control/display runs independently on a fixed 30Hz loop |
| 🤖 **Simulated Robot** | A Pymunk-based differential-drive robot with real inertia, turning, and physical collision — not a teleporting stub |
| 🟢 **Live Visualization** | Real-time overlay of the tracked person, goals, planned path, and the robot's actual (simulated) position |
| 📊 **Logging** | Logs the robot's actual simulated position and timestamp to CSV |

<br/>

## 🧩 Modular Architecture

```text
┌────────────┐
│  Webcam    │
└─────┬──────┘
      ↓
┌──────────────────────────────────────────────────────┐
│  Perception + Planning thread (own pace, ~3–4 FPS)    │
│                                                        │
│  Detect people/scene/goals/obstacles (single YOLO      │
│  pass + ByteTrack) → ground-plane homography →         │
│  A* over costmap → trapezoidal-velocity trajectory     │
└─────────────────────┬──────────────────────────────────┘
                       │  published via a lock-protected
                       │  SharedPerceptionState
                       ↓
┌──────────────────────────────────────────────────────┐
│  Control + display loop (fixed 30Hz, main thread)      │
│                                                        │
│  Sample trajectory by elapsed time → drive the         │
│  simulated robot (Pymunk) → log actual position →       │
│  draw overlay → cv2.imshow                              │
└────────────────────────────────────────────────────────┘
````

Perception/planning and control/display run at independent rates on separate threads, bridged by `SharedPerceptionState` — control keeps running smoothly even though detection only updates a few times a second.

<br/>

## 🚀 How to Run

```bash
git clone https://github.com/GKoutilya/smart-cam-navigator
cd smart-cam-navigator
pip install -r requirements.txt

# One-time ground-plane calibration (mark a rectangle on the floor,
# click its 4 corners in the webcam feed)
python scripts/calibrate_ground_plane.py

# Launch live demo
python main_pipeline.py
```

> Requires: Python 3.10+, OpenCV, Ultralytics YOLOv8, Pymunk, Matplotlib, NumPy

### Running the tests

```bash
python -m unittest discover -s tests -v
```

All tests are fully synthetic/mocked (fake cameras, fake YOLO results, a real-but-headless Pymunk simulation) — no webcam or model weights required to run them, and they run in CI on every push (see the badge above).

<br/>

## ⚙️ Tech Stack

- **Language**: Python 3.11
- **Vision**: OpenCV, Ultralytics YOLOv8 (with ByteTrack)
- **Planning**: A\* over a costmap, trapezoidal-velocity-profile trajectory generation
- **Simulation**: Pymunk (2D physics — matches the ground-plane fidelity level; no 3D engine needed)
- **Visualization**: Matplotlib, OpenCV overlays
- **Utils**: NumPy, CSV logging, threading (decoupled perception/control rates)

<br/>

## 💡 Future Extensions

* ✅ Ground-plane calibration, persistent tracking, obstacle-aware planning, decoupled rates, simulated differential-drive robot with a coupled wheel-speed model and physical collision — see the phase history in the module architecture above
* ⬜️ ROS2 interface
* ⬜️ Real robot deployment (e.g., Jetson Nano + differential-drive platform) — currently simulated only, by design (software-only project)
* ⬜️ LLM-based mission summaries (currently explored in [Project #5](https://github.com/GKoutilya/kitti-multisensor-perception-pipeline))

<br/>

## 🧠 Why This Matters

This project demonstrates my ability to:

* Build **real-time robotics pipelines** from scratch
* Combine **vision + planning + control**
* Design modular, **debuggable** and **extensible systems**
* Prioritize **frame rate and responsiveness**, not just ML accuracy
* Think like a **robotics software engineer**, not just an ML researcher

---

## 📬 Contact

**Koutilya Ganapathiraju**
* Machine Learning Engineer – Robotics
* Email: [gkoutilyaraju@gmail.com](mailto:gkoutilyaraju@gmail.com)
* GitHub: [GitHub](https://github.com/GKoutilya)
* LinkedIn: [LinkedIn](https://www.linkedin.com/in/koutilya-ganapathiraju-0a3350182/)

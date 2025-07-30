# 🧠 Real-Time Perception-to-Action Pipeline for Robotic Navigation

This project demonstrates a modular, real-time perception → planning → control pipeline designed for indoor robotic navigation using live webcam input. The system detects objects, infers scene context, estimates agent pose, plans dynamically updated paths, and simulates movement toward semantic goals — all with visual feedback and logging.

<br/>

## 🎯 Key Features

| Module | Description |
|--------|-------------|
| 🔍 **Perception** | YOLOv8 for object/person detection and custom scene type classification |
| 📍 **Pose Estimation** | Estimates agent (camera) position using detections as landmarks |
| 🧠 **Goal Inference** | Automatically chooses a semantic goal (e.g., exit, door, person) based on scene type |
| 🗺️ **Path Planning** | Plans shortest 2D path to the inferred goal using graph search or heuristic planning |
| 🔁 **Replanning** | Path updates every 1–2 seconds using background thread for scene analysis |
| 📹 **Camera Loop** | Live webcam input, stable frame rate (~3–4 FPS), play/pause support |
| 🟢 **Live Visualization** | Realtime overlay of pose, path, and goal on the camera feed (green = agent, red = path) |
| 📊 **Logging** | Logs scene type, detected objects, pose, and goal coordinates to CSV for review |

<br/>

## 🧩 Modular Architecture

```text
┌────────────┐
│  Webcam    │
└─────┬──────┘
      ↓
┌────────────┐
│ Perception │──┐  Detect objects, classify scene
└─────┬──────┘  │
      ↓         │
┌────────────┐  │
│  Pose Est. │◄─┘
└─────┬──────┘
      ↓
┌────────────┐
│ Goal Infer │   Choose semantic goal (e.g., person, door)
└─────┬──────┘
      ↓
┌────────────┐
│ Path Plan  │   Replans every 1–2s if needed
└─────┬──────┘
      ↓
┌────────────┐
│  Visualize │   Draw pose, goal, and path on live feed
└────────────┘
````

Each module is fully decoupled and interchangeable — ideal for future upgrades (e.g., replacing YOLO with a transformer-based detector, or plugging into ROS2).

<br/>

## 🖥️ Demo

| Live Webcam Feed                | Overlay with Pose + Path             |
| ------------------------------- | ------------------------------------ |
| ![camera](assets/frame_raw.jpg) | ![overlay](assets/frame_overlay.jpg) |

> ☑️ Try with different scenes: a hallway, a cluttered room, or a person walking into frame.

<br/>

## 🚀 How to Run

```bash
git clone https://github.com/yourusername/perception-to-action
cd perception-to-action
pip install -r requirements.txt

# Launch live demo
python main.py
```

> Requires: Python 3.10+, OpenCV, Ultralytics YOLOv8, Matplotlib, NumPy

<br/>

## 💡 Future Extensions

* ✅ Real robot integration (e.g., Jetson Nano + motors)
* ✅ Add motion smoothing and map memory
* ⬜️ ROS2 interface
* ⬜️ LLM-based mission summaries (currently explored in [Project #5](https://github.com/yourusername/kitti-fusion-gpt))

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
Machine Learning Engineer – Robotics
Email: [gkoutilyaraju@gmail.com](mailto:gkoutilyaraju@gmail.com)
GitHub: [GitHub](https://github.com/GKoutilya)
LinkedIn: [LinkedIn](https://www.linkedin.com/in/koutilya-ganapathiraju-0a3350182/)

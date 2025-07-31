# 🧠 Real-Time Perception-to-Action Pipeline for Robotic Navigation

This project demonstrates a modular, real-time perception → planning → control pipeline designed for indoor robotic navigation using live webcam input. The system detects objects, infers scene context, estimates agent pose, plans dynamically updated paths, and simulates movement toward semantic goals — all with visual feedback and logging.



## 🎯 Key Features

| Module | Description |
|--------|-------------|
| 🔍 **Perception** | YOLOv8 for object/person detection and custom scene type classification |
| 📍 **Pose Estimation** | Estimates agent (camera) position using detections as landmarks |
| 🧠 **Goal Inference** | Automatically selects a semantic goal (e.g., exit, door, person) based on scene type |
| 🗺️ **Path Planning** | Plans shortest 2D path to the goal using graph search or heuristics |
| 🔁 **Dynamic Replanning** | Periodically reprocesses frames for updated plans |
| 📹 **Camera Loop** | Live webcam input at ~3–4 FPS |
| 🟢 **Live Visualization** | Real-time overlay of pose, goal, and path (green = agent, red = path) |
| 📊 **Logging** | Logs timestamped pose and path coordinates to CSV |

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

## 🚀 How to Run

```bash
git clone https://github.com/GKoutilya/smart-cam-navigator
cd perception-to-action
pip install -r requirements.txt

# Launch live demo
python main_pipeline.py
```

> Requires: Python 3.10+, OpenCV, Ultralytics YOLOv8, Matplotlib, NumPy

<br/>

## ⚙️ Tech Stack

- **Language**: Python 3.11
- **Vision**: OpenCV, Ultralytics YOLOv8
- **Planning**: A\*, custom heuristic search
- **Visualization**: Matplotlib
- **Utils**: NumPy, CSV logging, threading

<br/>

## 💡 Future Extensions

* ✅ Real robot deployment (e.g., Jetson Nano + differential drive platform)
* ✅ Add motion smoothing and map memory
* ⬜️ ROS2 interface
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

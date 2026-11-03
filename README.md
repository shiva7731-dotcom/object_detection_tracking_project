# Object Detection and Tracking Project (Task 4)

This project fulfills **TASK 4: Object Detection and Tracking** using Python, `OpenCV`, and State-of-the-Art `YOLOv8` with built-in object tracking (ByteTrack/SORT).

---

## 📁 Project Structure
```text
object_detection_tracking_project/
├── main.py            # Main script for real-time video capture, YOLO detection, and tracking
├── requirements.txt   # Required Python packages
└── README.md          # Project guide and instructions
```

---

## 🛠️ Installation & Setup

1. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to Run

Run the tracking script from your terminal:
```bash
python main.py
```
*(By default, it uses your webcam `0`. To use a video file instead, change `0` in `main.py` to your video path, e.g., `"input_video.mp4"`).*

Press **`q`** on your keyboard to close the video feed window.

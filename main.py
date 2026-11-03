import cv2
import numpy as np
from ultralytics import YOLO

def run_object_detection_tracking(source="input_video.mp4"):
    """
    Runs real-time object detection and tracking using YOLOv8 and OpenCV.
    - Uses pre-trained YOLOv8n model for high-speed object detection.
    - Uses built-in ByteTrack/SORT tracking via YOLOv8 model.track().
    """
    # Load pre-trained YOLOv8 nano model (downloads automatically if not present)
    model = YOLO("yolov8n.pt")
    
    # Initialize video capture (0 for webcam, or path to video file)
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}'. Trying webcam (0)...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not access webcam either. Please check your video source.")
            return

    print("=" * 60)
    print("🎥 Object Detection & Tracking Active (Press 'q' to exit)")
    print("=" * 60)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or failed to grab frame.")
            break

        # Run YOLOv8 tracking on the frame (persist=True maintains tracking IDs across frames)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the output in a window
        cv2.imshow("YOLOv8 Real-Time Object Detection & Tracking", annotated_frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Tracking session ended.")

if __name__ == "__main__":
    # You can pass a video file path or use 0 for live webcam
    # Example: run_object_detection_tracking("input_video.mp4")
    run_object_detection_tracking(0)

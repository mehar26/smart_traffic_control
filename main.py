
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Run detection on a video
results = model.predict(
    source="trafficvideo.mp4",  # Path to video
    show=True,
    save=True,
    project="runs",
    name="video_output",
    exist_ok=True
)


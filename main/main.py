from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.yaml")
# model = YOLO("yolo11n.pt")
model = YOLO("ultralytics/cfg/models/11/yolo11.yaml").load("main/yolo11n.pt")

# Train the model
results = model.train(data="ultralytics/cfg/datasets/coco8.yaml", epochs=100, imgsz=640)
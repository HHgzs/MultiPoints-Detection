from ultralytics import YOLO


model = YOLO("ultralytics/cfg/models/11/yolo11-multipoints.yaml").load('runs/multipoints/multipoints/weights/last.pt')
results = model.export(format='onnx', imgsz=(480,640))  # export to onnx
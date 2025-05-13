from ultralytics import YOLO


# model = YOLO("yolo11n.yaml")
# model = YOLO("yolo11n.pt")

# model = YOLO("ultralytics/cfg/models/11/yolo11.yaml").load("main/yolo11n.pt")
# results = model.train(data="ultralytics/cfg/datasets/coco8.yaml", epochs=100, imgsz=640)


model = YOLO("ultralytics/cfg/models/11/yolo11-multipoints-rune-dist.yaml").load("main/yolo11n.pt")
results = model.train(data="ultralytics/cfg/datasets/rune_blender_mini.yaml", epochs=300, imgsz=640, name="rune")

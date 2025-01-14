from ultralytics import YOLO


image_path = 'datasets/armor/images/train/100210.jpg'
image_name = image_path.split('/')[-1].split('.')[0]

model = YOLO("ultralytics/cfg/models/11/yolo11-multipoints.yaml").load("runs/multipoints/multipoints/weights/last.pt")
result = model.predict(image_path, conf=0.1)
for r in result:
    r.save(f'./outputs/{image_name}.jpg')
    
from ultralytics import YOLO
import os
import random
import cv2
import numpy as np

n = 10
image_path = 'datasets/rune_blender/images/train'
output_folder = 'outputs/rune_blender'
background_folder = '/home/hhgzs/Project/MultiPoints-Detection/datasets/armor/images/train'
test_files = []

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if background_folder and os.path.exists(background_folder):
    background_files = [f for f in os.listdir(background_folder) if f.endswith('.png') or f.endswith('.jpg')]

def mix_background(img_path, background_path):
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    background = cv2.imread(background_path)

    if img is None:
        print(f"无法读取PNG图片: {img_path}")
        return None

    if background is None:
        print(f"无法读取背景图片: {background_files[0]}")
        return None

    # 调整背景图像大小与前景图像一致
    background = cv2.resize(background, (img.shape[1], img.shape[0]))

    # 分离前景图像的通道
    if img.shape[2] == 4:  # 确保前景图像有 alpha 通道
        b, g, r, alpha = cv2.split(img)
        alpha = alpha / 255.0  # 将 alpha 通道归一化到 [0, 1]
        
        # 使用 alpha 通道进行图像合成
        # foreground = cv2.merge((b, g, r))
        foreground = cv2.merge((r, g, b))
        for c in range(3):  # 对每个颜色通道进行加权合成
            background[:, :, c] = (alpha * foreground[:, :, c] + (1 - alpha) * background[:, :, c]).astype(np.uint8)
    else:
        print(f"前景图像缺少 alpha 通道: {img_path}")
        return None
    
    return background



def is_image_file(filename):
    return filename.lower().endswith(('.png', '.jpg', '.jpeg'))

if os.path.isfile(image_path) and is_image_file(image_path):
    test_files.append(image_path)
elif os.path.isdir(image_path):
    all_images = []
    for root, _, files in os.walk(image_path):
        for file in files:
            if is_image_file(file):
                all_images.append(os.path.join(root, file))
    test_files = random.sample(all_images, min(n, len(all_images)))

model = YOLO("ultralytics/cfg/models/11/yolo11-multipoints-rune.yaml").load("runs/multipoints/rune/weights/last.pt")
for image_file in test_files:
    if background_folder and os.path.exists(background_folder):
        background_file = random.choice(background_files)
        image = mix_background(image_file, os.path.join(background_folder, background_file))
    result = model.predict(image, conf=0.1)
    output_path = os.path.join(output_folder, os.path.basename(image_file))
    result[0].save(output_path)
    print(f"Processed {image_file} and saved to {output_path}")
    
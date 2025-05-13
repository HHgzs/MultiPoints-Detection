from ultralytics import YOLO
import os
import random
import cv2
import numpy as np

n = 5
image_path = 'datasets/rune_blender/images/train'
output_folder = 'outputs/rune_blender'
background_folder = 'datasets/armor/images/train'
model_path = 'runs/multipoints/rune/weights'
test_files = []

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if background_folder and os.path.exists(background_folder):
    background_files = [f for f in os.listdir(background_folder) if f.endswith('.png') or f.endswith('.jpg')]
model_files = [f for f in os.listdir(model_path) if f.endswith('.pt')]
while len(model_files) > 10:
    if len(model_files) % 2:
        model_files = model_files[1::2]
    else:
        model_files = model_files[::2]
model_lists = {}
for model_file in model_files:
    model_name = model_file.split('.')[0]
    if 'last' in model_file:
        continue
    if 'best' in model_file:
        continue
    model_lists[model_name] = YOLO("ultralytics/cfg/models/11/yolo11-multipoints-rune.yaml").load(os.path.join(model_path, model_file))
if len(model_lists) == 0:
    print("没有找到模型文件")
    exit(0)


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


for i, image_file in enumerate(test_files):
    if background_folder and os.path.exists(background_folder):
        background_file = random.choice(background_files)
        image = mix_background(image_file, os.path.join(background_folder, background_file))
        
    for j, (model_name, model) in enumerate(model_lists.items()):
        result = model.predict(image, conf=0.1)
        output_path = os.path.join(output_folder, f"image_{i+1}_{model_name}.jpg")
        result[0].save(output_path)
        print(f"Processed {image_file} and saved to {output_path}")
    
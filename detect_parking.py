import torch
import cv2
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Load trained model
model = torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/parking_detector/weights/best.pt', source='local')

# Image path
img_path = # replace with path

# Define custom save directory
output_dir = Path("yolov5/runs/detect/parking_detect")
output_dir.mkdir(parents=True, exist_ok=True)

# Run inference and save to custom dir
results = model(img_path)
results.save(save_dir=output_dir)

# Load result image
result_img_path = output_dir / os.path.basename(img_path)
img = cv2.imread(str(result_img_path))
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display
plt.imshow(img_rgb)
plt.axis('off')
plt.title("Detected Parking Slots")
plt.show()

import cv2
from ultralytics import YOLO

# 加载预训练的模型
model = YOLO('./drone/drone_train/weights/best.pt')

model.predict('datasets/predict/', project="drone",name="drone_predict",save=True,save_txt=True,save_conf=True, conf=0.25,exist_ok=True)
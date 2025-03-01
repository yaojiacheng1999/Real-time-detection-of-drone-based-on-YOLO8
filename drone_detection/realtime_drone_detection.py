import cv2
from ultralytics import YOLO

# 加载预训练的 YOLOv8 模型
model = YOLO('./drone/drone_train/weights/best.pt')

# 打开摄像头（默认摄像头，0为本机摄像头）
cap = cv2.VideoCapture(1)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 循环读取摄像头帧
while True:
    # 读取一帧
    ret, frame = cap.read()
    if not ret:
        print("无法读取帧")
        break

    # 使用 YOLOv8 模型进行预测
    results = model.predict(frame, conf=0.25)  # conf 是置信度阈值

    # 在帧上绘制检测结果
    annotated_frame = results[0].plot()  # 绘制检测框和标签

    # 显示带检测结果的帧
    cv2.imshow('Real-time Drone Detection', annotated_frame)

    # 按下 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
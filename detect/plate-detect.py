from ultralytics import YOLO

# 加载预训练的YOLOv8n模型
model = YOLO(r'F:\machinelearning\project\YOLOv8-LPRNet-car-plate-recognition\train\yolov8s.pt')
model = YOLO('../train/runs/detect/30000base-300epch/weights/best.pt')

# 在'bus.jpg'上运行推理，并附加参数
model.predict('F:/machinelearning/project/YOLOv8-LPRNet-car-plate-recognition/images-yolo/', save=True, imgsz=320, conf=0.5)
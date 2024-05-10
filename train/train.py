from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('yolov8s.pt')  # load a pretrained model (recommended for training)

# Train the model
model.train(data='F:/machinelearning/project/YOLOv8-LPRNet-car-plate-recognition/data/data.yaml', epochs=1, imgsz=640, batch=8, device=[0])
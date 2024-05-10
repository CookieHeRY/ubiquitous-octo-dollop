from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("../train/runs/detect/train12/weights/best.pt")  # load a pretrained model (recommended for training)

path = model.export(format="onnx")  # export the model to ONNX format
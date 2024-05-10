from ultralytics import YOLO
import torch

torch.cuda.empty_cache()
# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# # Use the model
model.train(data=r"F:\machinelearning\project\YOLOv8-LPRNet-car-plate-recognition\data\data.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("ultralytics/assets/bus.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format

# model = YOLO('yolov8n.pt')
# model.predict(source='ultralytics/assets/bus.jpg', save=True, imgsz=320, conf=0.5)

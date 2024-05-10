import os

import cv2

from ultralytics import YOLO


def traditional_iou(box1, box2):
    # 计算传统IOU
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = intersection / (area1 + area2 - intersection)
    return iou

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML
model = YOLO('../train/runs/detect/15000base-268epoch/weights/best.pt')

def getBox(filename):
    list1 = filename.split("-", 3)  # 第一次分割，以减号'-'做分割
    subname = list1[2]
    list2 = filename.split(".", 1)
    subname1 = list2[1]

    lt, rb = subname.split("_", 1)  # 第二次分割，以下划线'_'做分割
    lx, ly = lt.split("&", 1)
    rx, ry = rb.split("&", 1)

    boxes = [int(lx), int(rx), int(ly), int(ry)]

    return boxes

img_path = os.listdir('F:/machinelearning/project/datasets/test2/images')
num_train = len(img_path)
tp = 0
for i in range(num_train):
    fileName = os.path.join("F:/machinelearning/project/datasets/test2/images", img_path[i])
    img = cv2.imread(fileName)
    result = model.predict(fileName)
    result = result
    box1 = result[0].boxes.xyxy.cpu().numpy().reshape(-1)
    # print(box1)
    box2 = getBox(img_path[i])
    # print(box2)
    # print(traditional_iou(box1, box2))
    if(len(box2) < 4):
        continue
    if(len(box1) < 4):
        continue
    if(traditional_iou(box1, box2) >= 0.3):
        tp = tp + 1

print(tp)
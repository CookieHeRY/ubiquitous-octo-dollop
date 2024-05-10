import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
import sys
from model.LPRNet import build_lprnet
from data.load_data import CHARS

mean_value,std_value=(0.588,0.193)

def cv_imread(path):  #可以读取中文路径的图片
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            if temp.endswith('.jpg') or temp.endswith('.png') or temp.endswith('.JPG'):
                allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)

# 图片预处理
def transform(img):
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))

    return img

# 导入图片
def decodePlate(img, device):
    img = cv2.resize(img, (94, 24))
    im = transform(img)
    im = im[np.newaxis, :]
    ims = torch.Tensor(im)
    ims = ims.to(device)
    return ims

# 加载模型
def init_model(device, model_path):
    lprnet = build_lprnet(lpr_max_len=8, phase=True, dropout_rate=0.5)
    lprnet.to(device)
    print("Successful to build network!")
    lprnet.load_state_dict(torch.load(model_path))
    print("load pretrained model successfully!")
    lprnet.eval()
    return lprnet

def image_processing(img,device):
    img = cv2.resize(img, (168,48))
    img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img

# 得到车牌号
def get_plate_result(img, device, model):
    input = decodePlate(img, device)
    preds = model(input)
    preds = preds.cpu().detach().numpy()
    preb_labels = list()
    for i in range(preds.shape[0]):
        preb = preds[i, :, :]
        preb_label = list()
        for j in range(preb.shape[1]):
            preb_label.append(np.argmax(preb[:, j], axis=0))
        no_repeat_blank_label = list()
        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in preb_label:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        preb_labels.append(no_repeat_blank_label)

    plate = np.array(preb_labels)
    plate_result = ""
    for i in range(0, plate.shape[1]):
        b = CHARS[plate[0][i]]
        plate_result += b
    return plate_result
    # input = image_processing(img, device)
    # preds = model(input)
    # preds = torch.softmax(preds, dim=-1)
    # prob, index = preds.max(dim=-1)
    # index = index.view(-1).detach().cpu().numpy()
    # prob = prob.view(-1).detach().cpu().numpy()
    #
    # # preds=preds.view(-1).detach().cpu().numpy()
    # newPreds, new_index = decodePlate(index)
    # prob = prob[new_index]
    # plate = ""
    # for i in newPreds:
    #     plate += CHARS[i]
    # # if not (plate[0] in plateName[1:44] ):
    # #     return ""
    # return plate, prob

# 测试
if __name__ == '__main__':
    model_path = "../plate-recognition/weights/LPRNet__iteration_16000.pth"
    testPath = "../images-lpr"
    fileList = []
    allFilePath(testPath, fileList)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model(device, model_path)
    right = 0
    begin = time.time()

    for imge_path in fileList:
        img = cv_imread(imge_path)
        plate= get_plate_result(img, device, model)
        print(plate, imge_path)
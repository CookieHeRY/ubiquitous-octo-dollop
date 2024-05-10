import torch
import cv2
import numpy as np
from model.LPRNet import build_lprnet
from data.load_data import CHARS


# 图片预处理
def transform(img):
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))

    return img


# 导入一张图片
img = cv2.imdecode(np.fromfile('../images-lpr/川A0XN95.jpg', dtype=np.uint8), -1)
img = cv2.resize(img, (94, 24))
im = transform(img)
im = im[np.newaxis, :]
ims = torch.Tensor(im)

# 加载网络
lprnet = build_lprnet(lpr_max_len=8, phase=True, dropout_rate=0.5)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lprnet.to(device)
print("Successful to build network!")
lprnet.load_state_dict(torch.load("../train/weights/best_LPRNet_model.pth"))
print("load pretrained model successfully!")
print("皖AJ2572")

# 预测图片
prebs = lprnet(ims.to(device))
prebs = prebs.cpu().detach().numpy()
preb_labels = list()
for i in range(prebs.shape[0]):
    preb = prebs[i, :, :]
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

# 输出检测结果
plate = np.array(preb_labels)
a = ""
for i in range(0, plate.shape[1]):
    b = CHARS[plate[0][i]]
    a += b
print(a)

# ccpd数据集转为lpr识别格式
# 参考： https://blog.csdn.net/qq_36516958/article/details/114274778
#       https://github.com/HuKai97/YOLOv5-LPRNet-Licence-Recognition

import cv2
import os
import numpy as np
from PIL import Image

path = r'G:\CCPD2019\ccpd_base'  # 改成自己的车牌路径


provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']
num = 0
for filename in os.listdir(path):
    num += 1
    result = ""
    _, _, box, points, plate, brightness, blurriness = filename.split('-')
    list_plate = plate.split('_')  # 读取车牌
    result += provinces[int(list_plate[0])]
    result += alphabets[int(list_plate[1])]
    result += ads[int(list_plate[2])] + ads[int(list_plate[3])] + ads[int(list_plate[4])] + ads[int(list_plate[5])] + ads[int(list_plate[6])]
    # 新能源车牌的要求，如果不是新能源车牌可以删掉这个if
    # result += ads[int(list_plate[7])]
    # if result[2] != 'D' and result[2] != 'F' \
    #         and result[-1] != 'D' and result[-1] != 'F':
    #     print(filename)
    #     print("Error label, Please check!")
    #     # assert 0, "Error label ^~^!!!"
    #     continue
    # print(result)
    img_path = os.path.join(path, filename)
    img = cv2.imread(img_path)
    assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

    box = box.split('_')  # 车牌边界
    box = [list(map(int, i.split('&'))) for i in box]

    xmin = box[0][0]
    xmax = box[1][0]
    ymin = box[0][1]
    ymax = box[1][1]

    img = Image.fromarray(img)
    img = img.crop((xmin, ymin, xmax, ymax))  # 裁剪出车牌位置
    img = img.resize((94, 24), Image.LANCZOS)
    img = np.asarray(img)  # 转成array,变成24*94*3

    cv2.imencode('.jpg', img)[1].tofile(r"F:\machinelearning\project\datasets\ccpd\lpr\base\{}.jpg".format(result))
    # 图片中文名会报错
    # cv2.imwrite(r"F:\machinelearning\project\datasets\ccpd\lpr\test\test_remake\{}.jpg".format(result), img)
    if(num % 1000 == 0):
        print("已生成{}张".format(num))
print("共生成{}张".format(num))

#
import os
import random

import shutil
from shutil import copy2
trainfiles = os.listdir(r"F:\machinelearning\project\datasets\ccpd\lpr\base3")  #（图片文件夹）
num_train = len(trainfiles)
print("num_train: " + str(num_train) )
index_list = list(range(num_train))
print(index_list)
random.shuffle(index_list)  # 打乱顺序
num = 0
trainDir = r"F:\machinelearning\project\datasets\ccpd\lpr3\train"   #（将图片文件夹中的7份放在这个文件夹下）
validDir = r"F:\machinelearning\project\datasets\ccpd\lpr3\val"     #（将图片文件夹中的1份放在这个文件夹下）
detectDir = r"G:\yolo3\test\images"   #（将图片文件夹中的2份放在这个文件夹下）
for i in index_list:
    fileName = os.path.join(r"F:\machinelearning\project\datasets\ccpd\lpr\base3", trainfiles[i])  #（图片文件夹）+图片名=图片地址
    if num < num_train*0.8:  # yolo 7:1:2 lpr 8:2
        print(str(fileName))
        copy2(fileName, trainDir)
    # elif num < num_train*0.8:
    #     print(str(fileName))
    #     copy2(fileName, validDir)
    else:
        print(str(fileName))
        copy2(fileName, validDir)
    num += 1

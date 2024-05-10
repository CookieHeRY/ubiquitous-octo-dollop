# ccpd2019-base7000张图片，复杂场景（除无牌）各700张(可能存在重复)，以及ccpd2020的train部分的3500张图片作为训练集
# ccpd2019-base2000张图片，复杂场景（除无牌）各200张(可能存在重复)，以及ccpd2020的test部分的1000张图片作为测试集
# ccpd2019-base1000张图片，复杂场景（除无牌）各100张(可能存在重复)，以及ccpd2020的val部分的500张图片作为验证集
# 参考：https://blog.csdn.net/Suii_v5/article/details/72730792

import os, random, shutil


def copyFile(fileDir, tarDir):
    # 1
    pathDir = os.listdir(fileDir)

    # 2
    sample = random.sample(pathDir, 1000)

    # 3
    for name in sample:
        shutil.copyfile(fileDir + name, tarDir + name)


if __name__ == '__main__':
    # 源路径
    fileDir = "G:/CCPD2019/ccpd_base2/"
    # 目标路径
    tarDir = 'F:/machinelearning/project/datasets/test2/images/'
    copyFile(fileDir, tarDir)
    print("抽取完成")
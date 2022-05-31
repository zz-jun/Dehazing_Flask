import cv2
import numpy as np
import math

import torch
from PIL import Image
from torch.autograd import Variable


# def psnr1(img1, img2):
#     mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
#     if mse < 1.0e-10:
#         return 100
#     return 10 * math.log10(255.0**2/mse)
#
#
# def psnr2(img1, img2):
#     mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
#     if mse < 1.0e-10:
#         return 100
#     PIXEL_MAX = 1
#     return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


# psnr (Peak Signal to Noise Ratio, 峰值信噪比)
def getpsnr(origimal_img, filtered_img):
    '''
    计算图像的PSNR（Peak Signal to Noise Ratio）

    参数：
    -------
    origimal_img: 2-D array
                 图像的二维数组数据
    filtered_img: 2-D array
                图像的二维数组数据

    返回值：float，单位是dB
    ------
    返回psnr值
    '''
    if origimal_img.shape != filtered_img.shape:
        raise ValueError("the shapes of img1 and img2 are not same.")
    img1 = (origimal_img + 255) % 255  # 转到0-255
    img2 = (filtered_img + 255) % 255  # 转到0-255
    print("img shape: ",img1.shape)
    m,n,c = img1.shape
    mse = np.sum((img1 - img2) ** 2) / (m * n)  # 计算MSE
    return 20 * np.log10(np.max(img1) / np.sqrt(mse))  # 计算PSNR


if __name__ == '__main__':
    gt = cv2.imread('test/clear/1_2.jpg')
    # gt=Image.open('test/clear/1_2.jpg').convert('RGB')
    img = cv2.imread('test/psnr_ssim_img/FFA/pred_FFA_ots/img_4.png')

    # print(psnr1(gt, img))
    # print(psnr2(gt, img))
    print(getpsnr(gt, img))
    # print(getpsnr(img, gt))
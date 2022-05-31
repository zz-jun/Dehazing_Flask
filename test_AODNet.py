import glob

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from MyNet.AOD_Net import net
from MyNet.GCANet.utils import make_dataset
from PSNR import getpsnr
from SSIM import calculate_ssim

if __name__ == '__main__':

    # test_list = glob.glob("test_m/hazy/*")

    psnr_average = 0
    psnr_max = 0
    psnr_min = 100
    psnr_sum = 0
    num = 0
    ssim_average = 0
    ssim_max = 0
    ssim_min = 100
    ssim_sum = 0

    test_img_paths = make_dataset("test_m/hazy/")
    for image_path in test_img_paths:


        data_hazy = Image.open(image_path)
        data_hazy = data_hazy.convert('RGB')
        data_hazy = (np.asarray(data_hazy) / 255.0)

        data_hazy = torch.from_numpy(data_hazy).float()
        data_hazy = data_hazy.permute(2, 0, 1)
        data_hazy = data_hazy.cuda().unsqueeze(0)
        # data_hazy = data_hazy.unsqueeze(0)

        dehaze_net = net.dehaze_net().cuda()
        # dehaze_net = net.dehaze_net()
        # dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth'))
        dehaze_net.load_state_dict(torch.load('MyNet/AOD_Net/models/dehazer.pth', map_location="cuda"))

        clean_image = dehaze_net(data_hazy)
        # torchvision.utils.save_image(torch.cat((data_hazy, clean_image), 0), "test_m/AODNet/results/" + image_path.split("\\")[-1])
        torchvision.utils.save_image(clean_image, "test_m/AODNet/results/" + image_path.split("/")[-1])



        num=num+1
        img_Fall_name = image_path.split("/")[-1]
        img_name = img_Fall_name.split("_")[0]
        img_type = img_Fall_name.split(".")[-1]

        img_haze_path = "test_m/AODNet/results/"+img_Fall_name
        img_clear_path_n = "test_m/clear/" + img_name + ".png"

        gt = cv2.imread(img_clear_path_n)
        img = cv2.imread(img_haze_path)
        psnr=getpsnr(gt,img)

        psnr_sum = psnr_sum + psnr
        psnr_average = psnr_sum / num
        if psnr > psnr_max:
            psnr_max = psnr
        if psnr < psnr_min:
            psnr_min = psnr

        print("/*****************************************************************/")
        # print(img_Fall_name)
        # print(img_name)
        # print(img_type)
        # print(img_haze_path)
        # print(img_clear_path_n)

        print("PSNR:", psnr)
        print("psnr_max:", psnr_max)
        print("psnr_min:", psnr_min)
        print("psnr_average:", psnr_average)

        print("//////////////")

        ssim = calculate_ssim(gt, img)
        ssim_sum = ssim_sum + ssim
        ssim_average = ssim_sum / num
        if ssim > ssim_max:
            ssim_max = ssim
        if ssim < ssim_min:
            ssim_min = ssim
        print("ssim:", ssim)
        print("ssim_max:", ssim_max)
        print("ssim_min:", ssim_min)
        print("ssim_average:", ssim_average)

        print(image_path, "done!")

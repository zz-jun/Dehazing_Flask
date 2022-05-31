import PIL
import torch
import torchvision
import torch.optim
import numpy as np
from PIL import Image
import time
import glob

from torch import tensor
from torchvision import transforms

from GetNowTime import getNowTime
from MyNet.AOD_Net import net


def detect(image):
    print("*******AODNet*******")
    data_hazy = image
    data_hazy = data_hazy.convert('RGB')
    data_hazy = (np.asarray(data_hazy) / 255.0)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iscuda =  torch.cuda.is_available()
    print("device: ",device)
    print("iscuda: ",iscuda)


    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    if(iscuda):
        # 在第0个位置增加一个维度（batchsize维度）
        data_hazy = data_hazy.cuda().unsqueeze(0)
    else:
        # 在第0个位置增加一个维度（batchsize维度）
        data_hazy = data_hazy.unsqueeze(0)

    # print(data_hazy.shape)
    # print(data_hazy)

    if(iscuda):
        dehaze_net = net.dehaze_net().cuda()
    else:
        dehaze_net = net.dehaze_net()

    # dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth'))
    dehaze_net.load_state_dict(torch.load('MyNet/AOD_Net/models/dehazer.pth', map_location=device))

    clean_image = dehaze_net(data_hazy)
    # torchvision.utils.save_image(torch.cat((data_hazy, clean_image), 0), "output_image/AOD_Net/" + "AOD_Net.png")

    now_time = getNowTime()
    # 本地保存一张图片
    torchvision.utils.save_image(clean_image, "output_image/AOD_Net/" + "AOD_Net_" + now_time + ".png")

    # tensor 转换为PIL类型
    unloader = transforms.ToPILImage()
    clean_image = clean_image.cpu().clone()
    clean_image = clean_image.squeeze(0)
    clean_image = unloader(clean_image)


    # print(clean_image)
    print("AODNet image returning ...")
    return clean_image


if __name__ == '__main__':
    imgURL = r"test/haze/1909_0.85_0.2.jpg"
    image = Image.open(imgURL)
    img = detect(image)
    # img.show()
    print(img)

    # image = Image.open(imgURL)

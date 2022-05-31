import base64
import io
import os, argparse
import time

import numpy as np
from PIL import Image
from torchvision import transforms

from GetNowTime import getNowTime
from MyNet.FFA.FFA import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def tensorShow(tensors, titles=['haze']):
    fig = plt.figure()
    for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(221 + i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(tit)
    plt.show()


def detect(image):
    print("*******FFA*******")
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='ots', help='its or ots')
    opt = parser.parse_args()
    dataset = opt.task
    gps = 3
    blocks = 19

    # 输出目录
    abs = os.getcwd() + '/'
    output_dir = abs + f'output_image/FFA/pred_FFA_{dataset}/'
    # print("pred_dir:",output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model_dir = abs + f'MyNet/FFA/models/{dataset}_train_ffa_{gps}_{blocks}.pk'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: ",device)


    # print("model_dir:",model_dir)
    ckp = torch.load(model_dir, map_location=device)
    net = FFA(gps=gps, blocks=blocks)
    net = nn.DataParallel(net)
    net.load_state_dict(ckp['model'])
    net.eval()

    # print(image)
    haze = image
    haze = haze.convert('RGB')
    haze1 = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])(haze)[None, ::]
    haze_no = tfs.ToTensor()(haze)[None, ::]
    with torch.no_grad():
        pred = net(haze1)
    ts = torch.squeeze(pred.clamp(0, 1).cpu())

    now_time = getNowTime()
    vutils.save_image(ts, output_dir + "FFA_" + now_time + ".jpg")

    # print(ts.type())
    # tensor 转换为PIL类型
    unloader = transforms.ToPILImage()
    clean_image = ts.cpu().clone()
    clean_image = clean_image.squeeze(0)
    clean_image = unloader(clean_image)
    # print(clean_image)

    print("FFA image returning ...")
    return clean_image


if __name__ == '__main__':
    imgURL = r"test/haze/1909_0.85_0.2.jpg"
    image = Image.open(imgURL)
    # print(image)
    img = detect(image)
    # img.show()

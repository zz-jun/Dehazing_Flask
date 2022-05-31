import os,argparse

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from MyNet.FFA.FFA import FFA
from PSNR import getpsnr
from SSIM import calculate_ssim

abs=os.getcwd()+'/'
print(abs)

def tensorShow(tensors,titles=['haze']):
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='ots',help='its or ots')
parser.add_argument('--test_imgs',type=str,default='test_m/hazy',help='Test imgs folder')
opt=parser.parse_args()
dataset=opt.task
gps=3
blocks=19
img_dir=opt.test_imgs+'/'
output_dir=f'test_m/FFANet/output/'
print("pred_dir:",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_dir=abs+f'MyNet/FFA/models/{dataset}_train_ffa_{gps}_{blocks}.pk'
device='cuda' if torch.cuda.is_available() else 'cpu'
ckp=torch.load(model_dir,map_location=device)
net=FFA(gps=gps,blocks=blocks)
net=nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()

psnr_average = 0
psnr_max = 0
psnr_min = 100
psnr_sum = 0
num = 0
ssim_average=0
ssim_max=0
ssim_min=100
ssim_sum=0

for im in os.listdir(img_dir):
    # print(f'\r {im}',end='',flush=True)
    num=num+1
    print("/******************************************************************/")
    print(img_dir+im)
    haze = Image.open(img_dir+im)
    haze = haze.convert('RGB')
    haze1= tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])
    ])(haze)[None,::]
    haze_no=tfs.ToTensor()(haze)[None,::]
    with torch.no_grad():
        pred = net(haze1)
    ts=torch.squeeze(pred.clamp(0,1).cpu())
    # tensorShow([haze_no,pred.clamp(0,1).cpu()],['haze','pred'])
    # vutils.save_image(ts,output_dir+im.split('.')[0]+'_FFA.png')
    vutils.save_image(ts, output_dir + im.split('_')[0] + '.jpg')

    img_out=output_dir + im.split('_')[0] + ".jpg"

    img_clear="test_m/clear/"+im.split('_')[0]+".png"
    print(img_out)
    print(img_clear)

    gt = cv2.imread(img_clear)
    img = cv2.imread(img_out)
    psnr = getpsnr(gt, img)

    psnr_sum = psnr_sum + psnr
    psnr_average = psnr_sum / num
    if psnr > psnr_max:
        psnr_max = psnr
    if psnr < psnr_min:
        psnr_min = psnr

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



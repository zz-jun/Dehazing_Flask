import base64
import os
import argparse
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

from MyNet.GCANet.utils import make_dataset, edge_compute
from PSNR import getpsnr
from SSIM import calculate_ssim

parser = argparse.ArgumentParser()
parser.add_argument('--network', default='GCANet')
parser.add_argument('--task', default='dehaze', help='dehaze | derain')

# gpu_id < 0时，cpu可运行测试数据集
parser.add_argument('--gpu_id', type=int, default=0)
# parser.add_argument('--gpu_id', type=int, default=-1)

parser.add_argument('--indir', default='test_m/hazy/')
parser.add_argument('--outdir', default='test_m/GCANet/output')
opt = parser.parse_args()

# assert 条件为False时执行(opt.task的值不是'dehaze'和'derain'时报错提示)
assert opt.task in ['dehaze', 'derain']

## forget to regress the residue for deraining by mistake,
## which should be able to produce better results
opt.only_residual = opt.task == 'dehaze'
print("opt.only_residual  : ", opt.only_residual)
opt.model = 'MyNet/GCANet/models/wacv_gcanet_%s.pth' % opt.task
print("opt.model  : ", opt.model)

# opt.use_cuda = True/False。True时（gpu_id >= 0），GPU运行。FALSE时（gpu_id < 0）,CPU可运行
opt.use_cuda = opt.gpu_id >= 0

if not os.path.exists(opt.outdir):
    os.makedirs(opt.outdir)
test_img_paths = make_dataset(opt.indir)
# print(test_img_paths)

if opt.network == 'GCANet':
    from MyNet.GCANet.GCANet import GCANet

    net = GCANet(in_c=4, out_c=3, only_residual=opt.only_residual)
else:
    print('network structure %s not supported' % opt.network)
    raise ValueError

if opt.use_cuda:
    torch.cuda.set_device(opt.gpu_id)
    net.cuda()
else:
    net.float()

# GPU加载
# net.load_state_dict(torch.load(opt.model))
# CPU加载
net.load_state_dict(torch.load(opt.model, map_location='cuda'))
net.eval()

clear_path = "test_m/clear/"
out_path = "test_m/GCANet/output/"

psnr_average=0
psnr_max=0
psnr_min=100
psnr_sum=0
num=0
ssim_average=0
ssim_max=0
ssim_min=100
ssim_sum=0

for img_path in test_img_paths:
    print("/******************************************************************/")
    # print(img_path)
    num=num+1

    img_Fall_name = img_path.split("/")[-1]
    img_name = img_Fall_name.split("_")[0]
    img_type = img_Fall_name.split(".")[-1]

    img_haze_path = img_path
    img_clear_path_n = clear_path + img_name + ".png"
    # img
    # print(img_Fall_name)
    # print(img_name)
    # print(img_haze_path)
    # print(img_clear_path)

    img = Image.open(img_path).convert('RGB')
    im_w, im_h = img.size
    if im_w % 4 != 0 or im_h % 4 != 0:
        img = img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4)))
    img = np.array(img).astype('float')
    img_data = torch.from_numpy(img.transpose((2, 0, 1))).float()
    edge_data = edge_compute(img_data)
    in_data = torch.cat((img_data, edge_data), dim=0).unsqueeze(0) - 128
    in_data = in_data.cuda() if opt.use_cuda else in_data.float()
    with torch.no_grad():
        pred = net(Variable(in_data))
    if opt.only_residual:
        # 去雾图像 = 预测值 + 原图
        out_img_data = (pred.data[0].cpu().float() + img_data).round().clamp(0, 255)
    else:
        # 去雨图像 = 预测值
        out_img_data = pred.data[0].cpu().float().round().clamp(0, 255)
    # 保存图片
    out_img = Image.fromarray(out_img_data.numpy().astype(np.uint8).transpose(1, 2, 0))
    # out_img.save(os.path.join(opt.outdir, os.path.splitext(os.path.basename(img_path))[0] + '_%s.png' % opt.task))
    out_img.save(os.path.join(opt.outdir, os.path.splitext(os.path.basename(img_path))[0] + '.jpg'))

    out_p_n = out_path + img_Fall_name
    print(out_p_n)
    print(img_clear_path_n)

    # 图像大小调整
    gt = Image.open(img_path).convert('RGB')
    gt=gt.resize((int(im_w // 4 * 4), int(im_h // 4 * 4)))
    # PIL转base64
    img_buffer = BytesIO()
    gt.save(img_buffer, format='JPEG')
    byte_data = img_buffer.getvalue()
    base64_bytes = base64.b64encode(byte_data)
    # base64转ndarry
    img_str = base64.b64decode(base64_bytes)
    im_ndarray = np.fromstring(img_str, np.uint8)
    image = cv2.imdecode(im_ndarray, cv2.IMREAD_COLOR)  # BGR

    img = cv2.imread(out_p_n)

    # print(gt)
    # print(img.shape)
    PSNR=getpsnr(image,img)

    psnr_sum=psnr_sum+PSNR
    psnr_average=psnr_sum/num
    if PSNR>psnr_max:
        psnr_max=PSNR
    if PSNR<psnr_min:
        psnr_min=PSNR
    print("PSNR:",PSNR)
    print("psnr_max:", psnr_max)
    print("psnr_min:", psnr_min)
    print("psnr_average:", psnr_average)
    print("//////////////")

    ssim=calculate_ssim(image,img)
    ssim_sum=ssim_sum+ssim
    ssim_average=ssim_sum/num
    if ssim>ssim_max:
        ssim_max=ssim
    if ssim<ssim_min:
        ssim_min=ssim
    print("ssim:", ssim)
    print("ssim_max:", ssim_max)
    print("ssim_min:", ssim_min)
    print("ssim_average:", ssim_average)


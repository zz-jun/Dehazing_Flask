import os
import argparse
import time

import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

from GetNowTime import getNowTime
from MyNet.GCANet.utils import make_dataset, edge_compute


def detect(image):
    print("*******GCANet*******")
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', default='GCANet')
    parser.add_argument('--task', default='dehaze', help='dehaze | derain')

    # gpu_id < 0时，cpu可运行测试数据集
    # parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--gpu_id', type=int, default=0)

    # parser.add_argument('--indir', default='examples/')
    parser.add_argument('--outdir', default='output_image/GCANet')
    opt = parser.parse_args()

    # assert 条件为False时执行(opt.task的值不是'dehaze'和'derain'时报错提示)
    assert opt.task in ['dehaze', 'derain']

    ## forget to regress the residue for deraining by mistake,
    ## which should be able to produce better results
    opt.only_residual = opt.task == 'dehaze'
    # print("opt.only_residual  : ", opt.only_residual)
    opt.model = 'MyNet/GCANet/models/wacv_gcanet_%s.pth' % opt.task
    # print("opt.model  : ", opt.model)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iscuda = torch.cuda.is_available()
    print("device: ", device)
    print("iscuda: ", iscuda)

    # opt.use_cuda = True/False。True时（gpu_id >= 0），GPU运行。FALSE时（gpu_id < 0）,CPU可运行
    # opt.use_cuda = opt.gpu_id >= 0

    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)
    # test_img_paths = make_dataset(opt.indir)

    if opt.network == 'GCANet':
        from MyNet.GCANet.GCANet import GCANet
        net = GCANet(in_c=4, out_c=3, only_residual=opt.only_residual)
    else:
        print('network structure %s not supported' % opt.network)
        raise ValueError



    # if opt.use_cuda:
    if iscuda:
        torch.cuda.set_device(opt.gpu_id)
        net.cuda()
    else:
        net.float()


    net.load_state_dict(torch.load(opt.model, map_location=device))
    net.eval()
    # print(net)

    img = image.convert('RGB')
    im_w, im_h = img.size
    if im_w % 4 != 0 or im_h % 4 != 0:
        img = img.resize((int(im_w // 4 * 4), int(im_h // 4 * 4)))
    img = np.array(img).astype('float')
    img_data = torch.from_numpy(img.transpose((2, 0, 1))).float()
    edge_data = edge_compute(img_data)
    in_data = torch.cat((img_data, edge_data), dim=0).unsqueeze(0) - 128

    if iscuda:
        in_data = in_data.cuda()
    else:
        in_data.float()

    # in_data = in_data.cuda() if opt.use_cuda else in_data.float()


    with torch.no_grad():
        pred = net(Variable(in_data))
    if opt.only_residual:
        # 去雾图像 = 预测值 + 原图
        out_img_data = (pred.data[0].cpu().float() + img_data).round().clamp(0, 255)
    else:
        # 去雨图像 = 预测值
        out_img_data = pred.data[0].cpu().float().round().clamp(0, 255)


    # 保存图片
    now_time = getNowTime()
    out_img = Image.fromarray(out_img_data.numpy().astype(np.uint8).transpose(1, 2, 0))
    out_img.save(os.path.join(opt.outdir, "GCANet" + "_%s_" % opt.task + now_time + ".png"))


    print("GCANet image returning ...")
    return out_img


if __name__ == '__main__':
    imgURL = r"test/haze/1909_0.85_0.2.jpg"
    image = Image.open(imgURL)
    img = detect(image)
    # img.show()

import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm
from PIL import Image
from model import CSRNet
from my_dataset import create_test_dataloader
from utils import denormalize
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from matplotlib import cm as CM
import numpy as np

# 只计算出结果
def count1(img_root, model_param_path):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = CSRNet()
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    test_dataloader = create_test_dataloader(img_root)  # dataloader

    # 添加进度条
    for i, data in enumerate(tqdm(test_dataloader,ncols=50)):
        image = data['image'].to(device)
        et_densitymap = model(image).detach()
        # count = et_densitymap.data.sum()
        count = str('%.2f' % (et_densitymap[0].cpu().sum()))
        # print("renshu:", count)


# 保存结果文件
def count2(img_root, model_param_path):
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device=torch.device("cpu")
    model = CSRNet()
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    test_dataloader = create_test_dataloader(img_root)  # dataloader

    for i, data in enumerate(tqdm(test_dataloader,ncols=50)):
        image = data['image'].to(device)
        et_dmp = model(image).detach()
        # count = et_densitymap.data.sum()
        #count = str('%.2f' % (et_densitymap[0].cpu().sum()))
        # et_dmp = et_densitymap[0] / torch.max(et_densitymap[0])
        et_dmp = et_dmp.numpy()
        et_dmp = et_dmp[0][0]
        count=np.sum(et_dmp)
        plt.figure(i)
        plt.axis("off")
        plt.imshow(et_dmp,cmap=CM.jet)
        # 去除坐标轴
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # 输出图片边框设置
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(img_root + "/test_data/result/" + str(i + 1) + "_dmp" + ".jpg")
        print(str(i + 1) + "_"+"renshu:", count)


# 写入SummaryWriter
def count3(img_root, model_param_path):
    writer = SummaryWriter()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = CSRNet()
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    test_dataloader = create_test_dataloader(img_root)  # dataloader

    for i, data in enumerate(tqdm(test_dataloader,ncols=50)):
        image = data['image'].to(device)
        et_densitymap = model(image).detach()
        # count = et_densitymap.data.sum()
        count = str('%.2f' % (et_densitymap[0].cpu().sum()))
        writer.add_image(str(i) + '/img：', denormalize(image[0].cpu()))
        writer.add_image(str(i) + "/dmp_count:" + count, et_densitymap[0] / torch.max(et_densitymap[0]))
        print(str(i + 1) + "_img count success")



if __name__ == "__main__":
    img_root = "./data_test"
    model_param_path = "./checkpoints/checkpoint_A/best.pth"
    count2(img_root, model_param_path)

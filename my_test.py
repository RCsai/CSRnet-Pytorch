import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from torch.autograd import Variable
from tqdm import tqdm
from model import CSRNet
from PIL import Image
import os
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np
import cv2


def open_img(img_root):
    img = Image.open(img_root)
    if img.mode == 'L':
        # print('There is a grayscale image.')
        img = img.convert('RGB')
    return img


def one_count_A(img_path, model_param_path):
    if not os.path.exists('./result'):
        os.makedirs('./result')
    img_src_save_path = './result/test_src.jpg'
    img_et_save_path = './result/test_et.jpg'
    img_overlap_save_path = './result/test_overlap.jpg'
    filename = img_path.split('/')[-1]
    filenum = filename.split('.')[0]

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device=torch.device("cpu")
    model = CSRNet()
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    img_src = open_img(img_path)
    img = open_img(img_path)
    img_trans = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])])
    img = img_trans(img)
    img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    # print('img_src size:', img.shape)
    et_dmap = model(img)
    et_dmap = et_dmap.detach().numpy()
    et_dmap = et_dmap[0][0]
    people_num = np.sum(et_dmap)
    print(filenum + '_num:', '\t', people_num)

    # img_src
    plt.figure(0)
    plt.imshow(img_src)
    plt.axis('off')
    # 去除坐标轴
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # 输出图片边框设置
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(img_src_save_path, bbox_inches='tight', dpi=100, pad_inches=-0.04)

    # img_et
    plt.figure(1)
    plt.imshow(et_dmap, cmap=CM.jet)
    plt.axis('off')
    # 去除坐标轴
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # 输出图片边框设置
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(img_et_save_path, bbox_inches='tight', dpi=100, pad_inches=-0.04)
    # plt.show()

    # img_overlap
    img_src = cv2.imread(img_src_save_path)
    img_et = cv2.imread(img_et_save_path)
    img_et = cv2.resize(img_et, (img_src.shape[1], img_src.shape[0]))
    print('input_img_shape:',img_src.shape)
    print('output_img_shape:',img_et.shape)
    img_overlap = cv2.addWeighted(img_src, 0.2, img_et, 0.8, 0)
    cv2.imshow('img_overlap', img_overlap)
    cv2.imwrite(img_overlap_save_path, img_overlap)
    cv2.waitKey(0)


def one_count(img_path, model_param_path):
    filename = img_path.split('/')[-1]
    filenum = filename.split('.')[0]
    save_path = './data_test/test_data/result/' + filenum
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    img_src_save_path = save_path + '/' + filenum + '_src' + '.jpg'
    img_et_save_path = img_src_save_path.replace('src', 'et')
    img_overlap_save_path = img_src_save_path.replace('src', 'overlap')

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = CSRNet()
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    img_src = open_img(img_path)
    img = open_img(img_path)
    img_trans = Compose([ToTensor(), Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])])
    img = img_trans(img)
    img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False)
    # print('img_src size:', img.shape)
    et_dmap = model(img)
    et_dmap = et_dmap.detach().numpy()
    et_dmap = et_dmap[0][0]
    people_num = np.sum(et_dmap)
    # print(et_dmap.shape)
    print(filenum + '_num:', '\t', people_num)

    # img_src
    plt.figure(0)
    plt.imshow(img_src)
    plt.axis('off')
    # 去除坐标轴
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # 输出图片边框设置
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(img_src_save_path, bbox_inches='tight', dpi=100, pad_inches=-0.04)
    #
    # # img_et
    plt.figure(1)
    plt.imshow(et_dmap, cmap=CM.jet)
    plt.axis('off')
    # 去除坐标轴
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # 输出图片边框设置
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(img_et_save_path, bbox_inches='tight', dpi=100, pad_inches=-0.04)
    #
    img_src = cv2.imread(img_src_save_path)
    img_et = cv2.imread(img_et_save_path)
    img_et = cv2.resize(img_et, (img_src.shape[1], img_src.shape[0]))
    img_overlap = cv2.addWeighted(img_src, 0.2, img_et, 0.8, 0)
    cv2.imwrite(img_overlap_save_path, img_overlap)

    return people_num


def many_count(img_root, model_param_path):
    result_txt = './data_test/test_data/result/' + 'people_num.txt'
    f = open(result_txt, 'w')
    for filename in os.listdir(img_root):
        img_path = img_root + filename
        filenum = filename.split('.')[0]
        people_num = one_count(img_path, model_param_path)
        msg = filenum + ':' + str(people_num) + '\n'
        f.write(msg)
    f.close()
    print("count success")


def oneResizeImg(imgPath):
    imgName=imgPath.split('/')[-1]
    imgNum=imgName.split('.')[0]
    imgSavePath=imgPath.replace(imgNum,imgNum+'_re')
    img=cv2.imread(imgPath)
    img_resize=cv2.resize(img,(990,557))
    cv2.imwrite(imgSavePath,img_resize)

def manyResizeImg(imgroot):
    for filename in os.listdir(imgroot):
        imgPath=imgroot+filename
        oneResizeImg(imgPath)
    print("resize success")


# 计算一张
if __name__ == '__main__':
    img_path = './data_test/test_data/images/IMG_14.jpg'
    model_param_path = "./checkpoints/checkpoint_A/best.pth"
    one_count_A(img_path, model_param_path)

# # 计算多张
# if __name__ == "__main__":
#     img_root = "./data_test/test_data/images/"
#     model_param_path = "./checkpoints/checkpoint_A/best.pth"
#     many_count(img_root, model_param_path)

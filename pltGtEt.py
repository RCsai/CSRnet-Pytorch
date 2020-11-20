import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties


def getEtnum(et_num):
    et_txt_path = './result/txt/et_num_A.txt'
    f_et = open(et_txt_path)
    for line_txt in f_et:
        index_txt = line_txt.split(':')[0].split('_')[1]
        num_txt = line_txt.split(':')[1].replace('\n', '')
        index = int(index_txt) - 1
        num = round(float(num_txt))
        et_num[index] = num
    return et_num
    # print(index, '\t', num)


def getGtnum(gt_num):
    gt_txt_path = './result/txt/gt_num_A.txt'
    f_gt = open(gt_txt_path)
    for line_txt in f_gt:
        index_txt = line_txt.split(':')[0].split('_')[2]
        num_txt = line_txt.split(':')[1].replace('\n', '')
        index = int(index_txt) - 1
        num = round(float(num_txt))
        gt_num[index] = num
    return gt_num


def pltGtEt():
    NUMBER = 100  # 设置用于显示的数量
    x = np.zeros([NUMBER])
    et_num = np.zeros([182])
    gt_num = np.zeros([182])
    et_num = getEtnum(et_num)
    gt_num = getGtnum(gt_num)

    et = np.zeros([NUMBER])
    gt = np.zeros([NUMBER])

    for i in range(NUMBER):
        et[i] = et_num[i]
        gt[i] = gt_num[i]
    for i in range(NUMBER):
        x[i] = i
    font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)  # 设置字体样式
    plt.rcParams['font.sans-serif'] = ['Simhei']  # 解决中文显示问题，目前只知道黑体可行
    plt.figure('et_gt')
    plt.xlabel('图片id', fontproperties=font_set)
    plt.ylabel('人数', fontproperties=font_set)
    plt.plot(x, et, color='red', label='预测值')
    plt.plot(x, gt, color='blue', linestyle='--', label='实际值')
    plt.legend()  # 显示图例
    plt.show()


# 计算误差
def countError():
    NUMBER = 182  # 设置用于显示的数量
    et_num = np.zeros([182])
    gt_num = np.zeros([182])
    et_num = getEtnum(et_num)
    gt_num = getGtnum(gt_num)

    et = np.zeros([NUMBER])
    gt = np.zeros([NUMBER])
    error_pro = 0
    for i in range(NUMBER):
        et[i] = et_num[i]
        gt[i] = gt_num[i]
    error_num = np.zeros([5])
    for i in range(NUMBER):
        error = abs(et[i] - gt[i]) / gt[i]
        error_pro = error_pro + error
        if error < 0.1:
            error_num[0] += 1
        if error > 0.1 and error < 0.2:
            error_num[1] += 1
        if error > 0.2 and error < 0.3:
            error_num[2] += 1
        if error > 0.3 and error < 0.4:
            error_num[3] += 1
        if error > 0.4:
            error_num[4] += 1
    error_pro = error_pro / NUMBER
    print('错误等级次数:', error_num)
    print('平均错误概率:', error_pro)


def count_error():
    NUMBER = 100  # 设置用于显示的数量
    et_num = np.zeros([182])
    gt_num = np.zeros([182])
    et_num = getEtnum(et_num)
    gt_num = getGtnum(gt_num)
    num1 = 0
    num2 = 0
    NUMBER1=71
    NUMBER2=29
    error1=0
    error2=0
    y1 =np.zeros([NUMBER1])
    y2 = np.zeros([NUMBER2])
    z1 =np.zeros([NUMBER1])
    z2 = np.zeros([NUMBER2])
    x1 = np.zeros([NUMBER1])
    x2 = np.zeros([NUMBER2])

    for i in range(NUMBER):
        if gt_num[i] < 500:
            y1[num1] = gt_num[i]
            z1[num1] = et_num[i]
            num1 += 1
        else:
            y2[num2] = gt_num[i]
            z2[num2] = et_num[i]
            num2 += 1
    for i in range(num1):
        x1[i] = i
    for i in range(num2):
        x2[i] = i

    # print(num1,'\t',num2)

    for i in range(NUMBER1):
        error1=error1+abs(y1[i]-z1[i])/y1[i]
    error1=error1/NUMBER1
    for i in range(NUMBER2):
        error2=error2+abs(y2[i]-z2[i])/y2[i]
    error2=error2/NUMBER2

    print(error1)
    print(error2)

    font_set = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=12)  # 设置字体样式
    plt.rcParams['font.sans-serif'] = ['Simhei']  # 解决中文显示问题，目前只知道黑体可行
    plt.figure('<500')
    plt.xlabel('图片id', fontproperties=font_set)
    plt.ylabel('人数', fontproperties=font_set)
    plt.plot(x1, z1, color='red', label='预测值')
    plt.plot(x1, y1, color='blue', linestyle='--', label='实际值')
    plt.legend()  # 显示图例

    plt.figure('>500')
    plt.xlabel('图片id', fontproperties=font_set)
    plt.ylabel('人数', fontproperties=font_set)
    plt.plot(x2, z2, color='red', label='预测值')
    plt.plot(x2, y2, color='blue', linestyle='--', label='实际值')
    plt.legend()  # 显示图例

    plt.show()

def count_MAE_MSE():
    NUMBER = 182  # 设置用于显示的数量
    et_num = np.zeros([316])
    gt_num = np.zeros([316])
    et_num = getEtnum(et_num)
    gt_num = getGtnum(gt_num)
    mae = 0
    mse = 0
    et = np.zeros([NUMBER])
    gt = np.zeros([NUMBER])
    error_pro = 0
    for i in range(NUMBER):
        et[i] = et_num[i]
        gt[i] = gt_num[i]
    for i in range(NUMBER):
        mae = mae + abs(et[i] - gt[i])
        mse = mse + pow((et[i] - gt[i]), 2)
    mae = mae / NUMBER
    mse = np.sqrt(mse / NUMBER)
    print(mae)
    print(mse)


if __name__ == '__main__':
    # pltGtEt()
    # countError()
    # count_MAE_MSE()
    count_error()

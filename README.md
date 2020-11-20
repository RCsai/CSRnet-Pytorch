# CSRnet-Pytorch
人群计数算法CSRnet(使用深度学习框架Pytorch实现)

本人完成了ShanghaiTeach B数据集的训练工作

文件树形图如下：



文件说明：

checkpoint:   用于保存权重文件

data_test:    测试图片数据

data_train:   训练数据

result:       储存结果文件

run:          保存训练结果，用于可视化

config:       参数设置文件

dataset:      数据准备文件

gtCount:      用于计算测试数据真实值文件，并把结果保存为txt文件

model:        模型构建文件

mytest:       测试代码，保存图片结果，将预测人数保存为txt

pltGtEt:      根据真实值txt文件以及预测txt文件将两者在同一个坐标中进行显示

train:        训练文件


效果说明：

ShanghaiTeach B数据集计算结果为：MAE:10.8    MSE:16.0

图片展示：



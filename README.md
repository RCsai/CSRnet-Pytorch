# **CSRnet-Pytorch**



## 配置：

深度学习框架：Pytorch



## 文件说明：

checkpoint:		用于保存权重文件

data_test:			测试图片数据

data_train：		训练数据

result：				储存结果文件

run：					保存训练结果，用于可视化

config：				参数配置文件

dataset：			数据准备文件

gtCount：			用于计算测试数据真实值文件，并把结果保存为txt文件

model：				模型构建文件

mytest：				测试代码，保存图片结果，将预测人数保存为txt文件

pltGtEt：				根据真实值txt文件以及预测txt文件将两者在同一个坐标中进行显示

train：					训练文件

data_train/mall_dataset_A/dmap_for_SHHA.py:  对图片进行预处理，生成densitymaps文件夹中文件

data_train/mall_dataset_B/dmap_for_SHHB.py:  对图片进行预处理，生成densitymaps文件夹中文件



## 训练结果

测试结果：

| 数据集          | MAE  | MSE  |
| --------------- | ---- | ---- |
| ShanghaiTeach B | 10.8 | 16.0 |



模型效果：


![演示效果图片](https://github.com/RCsai/CSRnet-Pytorch/blob/main/result/txt/show.jpg)

import torch
import os
from tensorboardX import SummaryWriter


class Config():
    '''
    Config class
    '''

    def __init__(self):
        # self.dataset_root = './data_train_test/mall_dataset_A'
        self.dataset_root = './data_train/mall_dataset_B'

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.lr = 1e-5  # learning rate
        self.batch_size = 1  # batch size
        # self.epochs       = 2000                # epochs
        self.epochs = 200
        self.checkpoints = './checkpoints/checkpoint_B'  # checkpoints dir
        self.writer = SummaryWriter()  # tensorboard writer

        self.__mkdir(self.checkpoints)

    def __mkdir(self, path):
        '''
        create directory while not exist
        '''
        if not os.path.exists(path):
            os.makedirs(path)
            print('create dir: ', path)

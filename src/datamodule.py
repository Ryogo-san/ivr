import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

import dataset
from dataset import MyDataset

import utils
from utils import *
import config
from config import CFG

class MyDataModule(LightningDataModule):
    def __init__(self, img_path_list, cfg):
        super().__init__()
        self.train_list=img_path_list # 
        self.val_df = img_path_list # 
        self.cfg = cfg

    def __create_dataset(self, train=True):
        if train:
            return MyDataset(self.train_list, self.cfg.image_size,train)
        else:
            return MyDataset(self.val_df, self.cfg.image_size,train)

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        #index=0
        #print(dataset.__getitem__(index)[1])
        return DataLoader(dataset, self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)


if __name__=="__main__":
    img_list=get_image_path_list()
    module=MyDataModule(img_list,CFG)
    loader=module.train_dataloader()
    batch_iterator=iter(loader)
    inputs,labels=next(batch_iterator)
    print(inputs.size())
    print(labels.size())

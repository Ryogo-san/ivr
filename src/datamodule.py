import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader,random_split
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
        num_of_img_path=len(img_path_list)
        train_size=int(0.9*num_of_img_path)
        val_size=num_of_img_path-train_size
        self.train_list,self.val_list=random_split(img_path_list,[train_size,val_size])
        self.cfg = cfg

    def __create_dataset(self, train=True):
        if train:
            return MyDataset(self.train_list, self.cfg.image_size)
        else:
            return MyDataset(self.val_list, self.cfg.image_size)

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()


if __name__=="__main__":
    img_list=get_image_path_list(CFG.data_dir)
    module=MyDataModule(img_list,CFG)
    loader=module.test_dataloader()
    batch_iterator=iter(loader)
    inputs,labels=next(batch_iterator)
    print(inputs.size())
    print(labels.size())

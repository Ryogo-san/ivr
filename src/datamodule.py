import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule

import dataset
from dataset import MyDataset


class MyDataModule(LightningDataModule):
    def __init__(self, train_df, val_df, cfg):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.cfg = cfg

    def __create_dataset(self, train=True):
        if train:
            return MyDataset(self.train_df, self.cfg.image_size)
        else:
            return MyDataset(self.val_df, self.cfg.image_size)

    def train_dataloader(self):
        dataset = self.__create_dataset(True)
        return DataLoader(dataset, self.cfg.batch_size, shuffle=True, num_workers=self.cfg.num_workers)

    def val_dataloader(self):
        dataset = self.__create_dataset(False)
        return DataLoader(dataset, self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)

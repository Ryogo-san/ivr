import os

import numpy as np
import pytorch_lightning as pl
import torch

import datamodule
from datamodule import MyDataModule
import model
from model import RecognitionModel
import config
from  config import CFG
import utils
from utils import *


def main(img_list, cfg):
    model=RecognitionModel(cfg)
    data=MyDataModule(img_list,cfg)

    trainer=pl.Trainer()
    trainer.fit(model,data)


if __name__=="__main__":
    img_list=get_image_path_list()
    main(img_list,CFG)

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

import datamodule
from datamodule import MyDataModule
import model
from model import RecognitionModel
import callbacks
from callbacks import *
import config
from  config import CFG
import utils
from utils import *


def main(img_list, cfg):
    model=RecognitionModel(cfg)
    data=MyDataModule(img_list,cfg)


    ckpt_path="./lightning_logs/version_29/checkpoints/epoch=14-step=4199.ckpt"
    state_dict=torch.load(ckpt_path)
    model=model.load_from_checkpoint(checkpoint_path=ckpt_path,cfg=CFG)
    trainer=pl.Trainer()
    trainer.test(model,data)

if __name__=="__main__":
    img_list=get_image_path_list(CFG.data_dir)
    main(img_list,CFG)

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

    early_stop_callback=EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
    )

    my_callback=MyCallback()

    checkpoint_callback=ModelCheckpoint(
            dirpath="../model",
            filename="best_model"
    )

    trainer=pl.Trainer(
            max_epochs=CFG.epochs,
            gpus=1,
            callbacks=[my_callback,early_stop_callback,checkpoint_callback]
    )

    trainer.fit(model,data)


if __name__=="__main__":
    img_list=get_image_path_list()
    main(img_list,CFG)

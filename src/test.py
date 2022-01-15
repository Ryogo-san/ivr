import os
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
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
from config import CFG
import utils
from utils import *


def main(img_list, cfg):
    seed_everything(cfg.seed)

    model = RecognitionModel(cfg)
    data = MyDataModule(img_list, cfg)

    ckpt_path = CFG.ckpt_path
    state_dict = torch.load(ckpt_path,map_location=torch.device("cpu"))
    model = model.load_from_checkpoint(checkpoint_path=ckpt_path, cfg=cfg)
    trainer = pl.Trainer(deterministic=True)
    trainer.test(model, data)


if __name__ == "__main__":
    img_list = get_image_path_list(CFG.data_dir)
    main(img_list, CFG)

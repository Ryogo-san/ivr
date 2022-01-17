import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
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
    seed_everything(seed=CFG.seed)

    model = RecognitionModel(cfg)
    data = MyDataModule(img_list, cfg)

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=cfg.patience,
    )

    my_callback = MyCallback()

    checkpoint_callback=ModelCheckpoint(
            dirpath=cfg.model_dir,
            filename="best_model"
    )

    bar = MyProgressBar(refresh_rate=5, process_position=1)

    logger = TensorBoardLogger(
            save_dir=cfg.output_dir,
            name="logs"
    )

    trainer = pl.Trainer(
        max_epochs=CFG.epochs,
        gpus=CFG.gpus,
        callbacks=[my_callback, early_stop_callback, bar,checkpoint_callback],
        deterministic=True,
        logger=logger
    )

    trainer.fit(model, data)


if __name__ == "__main__":
    img_list = get_image_path_list(CFG.data_dir)
    main(img_list, CFG)

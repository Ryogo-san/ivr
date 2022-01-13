import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ProgressBar


class MyCallback(Callback):
    def on_init_start(self,trainer):
        print("Starting to init trainer!")

    def on_epoch_end(self,trainer,pl_module):
        print("")


class MyProgressBar(ProgressBar):
    def init_validation_tqdm(self):
        bar=super().init_validation_tqdm()
        bar.set_description("running validation ...")
        return bar

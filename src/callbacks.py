import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class MyCallback(Callback):
    def on_init_start(self,trainer):
        print("Starting to init trainer!")



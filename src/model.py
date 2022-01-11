import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy

import utils
from utils import *


class RecognitionModel(LightningModule):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.accuracy = Accuracy()
        self.model = timm.create_model(self.cfg.model_name, pretrained=pretrained)

        if self.cfg.model_name in self.cfg.fc_models:
            self.n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()
            self.fc = nn.Linear(self.n_features, self.cfg.target_size)

    def get_feature(self, image):
        feature = self.model(image)
        return feature

    def forward(self, image):
        feature = self.get_feature(image)
        output = self.fc(feature)
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y=y.long()
        y_hat=y_hat.long()
        loss = get_loss(y_hat, y,self.cfg)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y=y.long()
        y_hat=y_hat.long()
        loss = get_loss(y_hat, y,self.cfg)
        preds = torch.argmax(loss, dim=1)  #
        self.accuracy(preds, y)  #

    def test_step(self, batch, batch_idx):
        return validation_step(self, batch, batch_idx)

    def configure_optimizers(self):
        optimizer = get_optimizer(self.cfg.optimizer,self.parameters(),self.cfg.learning_rate)
        scheduler = get_scheduler(optimizer,self.cfg)

        return [optimizer], [scheduler]


if __name__ == "__main__":
    model = RecognitionModel(CFG)
    print(model)

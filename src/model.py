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
        self.input_conv=nn.Conv2d(1,3,kernel_size=1)

        if self.cfg.model_name in self.cfg.classifier_models:
            self.n_features = self.model.classifier.in_features
            self.model.classifier = nn.Identity()

        elif self.cfg.model_name in self.cfg.fc_models:
            self.n_features = self.model.fc.in_features
            self.model.fc = nn.Identity()

        self.fc1 = nn.Linear(self.n_features, self.cfg.num_method_classes)
        self.fc2 = nn.Linear(self.n_features, self.cfg.num_letter_classes)
        self.softmax = nn.Softmax()

    def get_feature(self, image):
        feature = self.model(image)
        return feature

    def forward(self, image):
        x=self.input_conv(image)
        feature = self.get_feature(x)
        out_method = self.softmax(self.fc1(feature))
        out_letter = self.softmax(self.fc2(feature))
        return out_method, out_letter

    def training_step(self, batch, batch_idx):
        x, y = batch
        out_method, out_letter = self(x)
        loss = self.cfg.alpha * get_loss(out_method, y[:, 0], self.cfg) + (1 - self.cfg.alpha) * get_loss(out_letter, y[:, 1], self.cfg)
        self.log("train_loss",loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out_method, out_letter = self(x)
        loss1 = get_loss(out_method, y[:, 0], self.cfg)
        loss2 = get_loss(out_letter, y[:, 1], self.cfg)
        loss = self.cfg.alpha * loss1 + (1 - self.cfg.alpha) * loss2
        preds_method = torch.argmax(out_method, dim=1)  #
        preds_letter = torch.argmax(out_letter, dim=1)
        method_acc = self.accuracy(preds_method, y[:, 0])  #
        letter_acc = self.accuracy(preds_letter, y[:, 1])  #
        self.log("val_loss", loss)
        self.log("method_acc", method_acc)
        self.log("letter_acc", letter_acc)

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = get_optimizer(
            self.cfg.optimizer, self.parameters(), self.cfg.learning_rate
        )
        scheduler = get_scheduler(optimizer, self.cfg)

        return [optimizer], [scheduler]


if __name__ == "__main__":
    model = RecognitionModel(CFG)
    print(model)

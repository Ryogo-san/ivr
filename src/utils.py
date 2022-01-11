import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

import config
from config import CFG


def get_optimizer(optimizer_name, parameters, learning_rate):
    """optimizer getter

    optimizerをCFGにしたがってconfigure_optimizersに渡す関数

    Args:
        optimizer_name (str): the name of optimizer you want to use.
        parameters: self.parameters用
        learning_rate: 学習率

    Returns:
        optimizer
    """
    if optimizer_name == "Adam":
        optimizer = Adam(parameters, learning_rate)

    return optimizer


def get_scheduler(optimizer, cfg):
    if cfg.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr, last_epoch=-1)

    return scheduler


def get_image_path_list():
    train_list=[]
    for dirname,_,filenames in os.walk("../data/dataset"):
        for filename in filenames:
            train_list.append(os.path.join(dirname,filename))

    return train_list

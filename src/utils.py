import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        scheduler = CosineAnnealingLR(
            optimizer, T_max=cfg.T_max, eta_min=cfg.min_lr, last_epoch=-1
        )

    return scheduler


def get_loss(yhat, y, cfg):
    if cfg.loss == "CE":
        return F.cross_entropy(yhat, y)

def calculate_l1_loss(parameters):
    l1=torch.tensor(0.,requires_grad=True)
    for w in parameters:
        l1=l1+torch.norm(w,1)
    return l1

def get_image_path_list(data_dir):
    train_list = []
    for dirname, _, filenames in os.walk(data_dir):
        for filename in filenames:
            train_list.append(os.path.join(dirname, filename))

    return train_list


def hiragana_idx_to_unicode(idx):
    hiragana_unicode = [
        "3042", "3044", "3046", "3048", "304A",
        "304B", "304D", "304F", "3051", "3053",
        "3055", "3057", "3059", "305B", "305D",
        "305F", "3061", "3064", "3066", "3068",
        "306A", "306B", "306C", "306D", "306E",
        "306F", "3072", "3075", "3078", "307B",
        "307E", "307F", "3080", "3081", "3082",
        "3084", "3086", "3088", 
        "3089", "308A", "308B", "308C", "308D",
        "308F", "3092", "3093",
    ]

    return "U" + hiragana_unicode[idx]


def unicode_to_hiragana_idx(unicode_label):
    code = unicode_label[1:]  # U3044 -> 3044
    hiragana_unicode = [
        "3042", "3044", "3046", "3048", "304A",
        "304B", "304D", "304F", "3051", "3053",
        "3055", "3057", "3059", "305B", "305D",
        "305F", "3061", "3064", "3066", "3068",
        "306A", "306B", "306C", "306D", "306E",
        "306F", "3072", "3075", "3078", "307B",
        "307E", "307F", "3080", "3081", "3082",
        "3084", "3086", "3088", 
        "3089", "308A", "308B", "308C", "308D",
        "308F", "3092", "3093",
    ]

    return hiragana_unicode.index(code)


def get_label_method_class(label_method):
    if label_method == "ball-pen":
        return 0
    elif label_method == "enpitu":
        return 1
    elif label_method == "sya-pen":
        return 2

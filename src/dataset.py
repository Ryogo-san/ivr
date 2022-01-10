import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    """dataset"""

    def __init__(self, df, image_size, transform, train=True):
        self.x = df
        self.y = None
        if train:
            self.y = df[y]
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        pass

import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

import utils
from utils import *


class MyDataset(Dataset):
    """dataset"""

    def __init__(self, file_list, image_size):
        self.file_list=file_list
        self.transform = transforms.Compose([transforms.ToTensor()]) #

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path=self.file_list[idx]
        img=cv2.imread(img_path)
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        img_transformed=self.transform(img) #
        img_path_list=img_path.split("/") # ./enpitu/U3042/U3042_00000.png -> enpitu, U3042
        label_method=img_path_list[3] # enpitu
        label_letter=img_path_list[4] # U3042

        label_method=get_label_method_class(label_method)
        label_letter=unicode_to_hiragana_idx(label_letter)

        return img_transformed,torch.tensor([label_method,label_letter])


if __name__=="__main__":
    train_list=get_image_path_list()
    train_dataset=MyDataset(train_list,image_size=256)
    index=0
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])

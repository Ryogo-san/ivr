import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms


class MyDataset(Dataset):
    """dataset"""

    def __init__(self, file_list, image_size, transform, train=True):
        self.file_list=file_list
        self.transform = transforms.Compose([transforms.ToTensor()]) #

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path=self.file_list[idx]
        img=cv2.imread(img_path)

        img_transformed=self.transform(img) #
        print(type(img_transformed))
        img_path_list=img_path.split("/") # ./enpitu/U3042/U3042_00000.png -> enpitu, U3042
        label_method=img_path_list[3] # enpitu
        label_letter=img_path_list[4] # U3042

        if label_method=="ball_pen":
            label_method=0
        elif label_method=="enpitu":
            label_method=1
        elif label_method=="sya-pen":
            label_method=2


        if label_letter=="U3042":
            label_letter=0
        else:
            label_letter=1
        return img_transformed,torch.tensor([label_method,label_letter])


if __name__=="__main__":
    
    train_list=[]
    for dirname,_,filenames in os.walk("../data/dataset"):
        for filename in filenames:
            train_list.append(os.path.join(dirname,filename))

    train_dataset=MyDataset(train_list,image_size=256,transform=None,train=True)
    index=0
    print(train_dataset.__getitem__(index)[0].size())
    print(train_dataset.__getitem__(index)[1])

import numpy as np
import os
import cv2
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.dirname(__file__),"../"))
import argparse

from pytorch_lightning.utilities.seed import seed_everything

import model
from model import *
import config
from config import CFG
import datamodule
from datamodule import MyDataModule
import fgsm
from fgsm import FGSM
import pgd
from pgd import PGD
import attack_dispatcher
from attack_dispatcher import attack_dispatcher
import utils
from utils import *


def imsave(img,img_file_name):
        plt.imshow(img,cmap="gray")
        plt.axis("off")
        plt.savefig(img_file_name)
        plt.clf()

def main(img_list,cfg,device,attack_name):
    seed_everything(seed=cfg.seed)
    model=RecognitionModel(cfg)
    module=MyDataModule(img_list,cfg)

    ckpt_path=CFG.ckpt_path
    state_dict=torch.load(ckpt_path,map_location=torch.device("cpu"))
    model=model.load_from_checkpoint(checkpoint_path=ckpt_path,cfg=cfg)
    
    model=model.eval()
    
    attack_input={
            "model":model,
            "device":device,
            "cfg":cfg
    }

    attack=attack_dispatcher(attack_name,**attack_input)

    correct_method=0
    correct_letter=0
    total=0

    loader = module.test_dataloader()

    for images,labels in loader:
        adv_images=attack(images,labels)
        methods,letters=model(adv_images)

        pre_method=torch.argmax(methods,dim=1)
        pre_letter=torch.argmax(letters,dim=1)

        correct_method+=(pre_method==labels[:,0]).sum()
        correct_letter+=(pre_letter==labels[:,1]).sum()
        result_img=adv_images[0].permute(1,2,0).numpy()

        figname_tag=int(total/cfg.batch_size)

        os.makedirs(f"./adversarials/{attack_name}/",exist_ok=True)
        imsave(result_img,f"./adversarials/{attack_name}/adv_{figname_tag}.png")
        
        input_img=images[0].permute(1,2,0).detach().numpy()
        imsave(input_img,f"./adversarials/{attack_name}/input_{figname_tag}.png")

        # show the result
        print(f"pred_pen: {pre_method[0]}, answer_pen: {labels[0,0]}")
        print(f"pred_letter: {pre_letter[0]}, answer_letter: {labels[0,1]}")
        print("images saved\n")
        total+=CFG.batch_size

    print("Accuracy of method: %.2f%%"%(100*float(correct_method)/total))
    print("Accuracy of letter: %.2f%%"%(100*float(correct_letter)/total))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--attack",default="FGSM")
    args=parser.parse_args()

    img_list = get_image_path_list(CFG.data_dir)
    device="cuda:0" if torch.cuda.is_available() else "cpu"
    main(img_list,CFG,device,args.attack)

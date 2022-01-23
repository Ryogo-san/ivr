import torch
import torch.nn as nn

import attack
from attack import Attack


class FGSM(Attack):
    def __init__(self, model, device, cfg, eps=0.007):
        super().__init__(model)
        self.cfg = cfg
        self.eps = self.cfg.pert_eps
        self.device = device

    def forward(self, images, labels):
        images = images
        labels = labels
        loss_func = nn.CrossEntropyLoss()

        images.requires_grad = True
        methods, letters = self.model(images)
        loss1 = loss_func(methods, labels[:, 0])
        loss2 = loss_func(letters, labels[:, 1])
        loss = self.cfg.alpha * loss1 + (1 - self.cfg.alpha) * loss2

        grad = torch.autograd.grad(
            loss, images, retain_graph=False, create_graph=False
        )[0]
        adv_images = images + self.eps * grad.sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images

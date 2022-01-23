import os
import torch
import torch.nn as nn
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import utils
from utils import calculate_l1_loss
import attack
from attack import Attack


class MIM(Attack):
    def __init__(
        self, model, device, cfg, steps=5, eps=8 / 255, alpha=2 / 255, decay=1.0
    ):
        super().__init__(model)
        self.cfg = cfg
        self.device = device
        self.steps = steps
        self.eps = eps
        self.alpha = alpha
        self.decay = decay

    def forward(self, images, labels):
        images = images.clone().detach()
        labels = labels.clone().detach()
        loss_func = nn.CrossEntropyLoss()

        momentum = torch.zeros_like(images).detach()

        adv_images = images.clone().detach()

        for _ in range(self.steps):
            adv_images.requires_grad = True
            pens, characters = self.model(adv_images)

            loss1 = loss_func(pens, labels[:, 0]).to(self.device)
            loss2 = loss_func(characters, labels[:, 1]).to(self.device)
            loss = self.cfg.alpha * loss1 + (1 - self.cfg.alpha) * loss2

            grad = torch.autograd.grad(
                loss,
                adv_images,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
            grad = grad + momentum * self.decay
            momentum = grad
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

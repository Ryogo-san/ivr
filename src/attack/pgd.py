import torch
import torch.nn as nn

import attack
from attack import Attack


class PGD(Attack):
    def __init__(self, model, device, cfg, steps=5, eps=8 / 255, alpha=2 / 255):
        super().__init__(model)
        self.cfg = cfg
        self.device = device
        self.steps = steps
        self.eps = eps
        self.alpha = alpha

    def forward(self, images, labels):
        images = images.clone().detach()
        labels = labels.clone().detach()
        loss_func = nn.CrossEntropyLoss()

        adv_images = images.clone().detach()

        #  random start
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(
            -self.eps, self.eps
        )
        adv_images = torch.clamp(adv_images, min=0, max=1)

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
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images

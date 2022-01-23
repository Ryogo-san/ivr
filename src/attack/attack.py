import torch


class Attack(object):
    def __init__(self, model):

        self.model = model

    def forward(self, *input):
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images

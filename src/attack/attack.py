import torch


class Attack(object):
    r"""
    Base class for all attacks.
    .. note::
        It automatically set device to the device where given model is.
        It temporarily changes the original model's `training mode` to `test`
        by `.eval()` only during an attack process.
    """

    def __init__(self, model):
        r"""
        Initializes internal Attack state.
        Arguments:
            name (str) : name of attack.
            model (nn.Module): model to attack.
        """

        self.model = model

    # It defines the computation performed at every call.
    # Should be overridden by all subclasses.
    def forward(self, *input):
        r"""
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError

    def __call__(self, *input, **kwargs):
        images = self.forward(*input, **kwargs)
        return images

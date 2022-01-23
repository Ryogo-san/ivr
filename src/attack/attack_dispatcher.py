import fgsm
from fgsm import FGSM
import pgd
from pgd import PGD
import mim
from mim import MIM


def attack_dispatcher(name, *args, **kwargs):
    if name == "FGSM":
        return FGSM(*args, **kwargs)
    elif name == "PGD":
        return PGD(*args, **kwargs)
    elif name == "MIM":
        return MIM(*args, **kwargs)

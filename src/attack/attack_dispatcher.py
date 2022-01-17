import fgsm
from fgsm import FGSM
import pgd
from pgd import PGD


def attack_dispatcher(name,*args,**kwargs):
    if name=="FGSM":
        return FGSM(*args,**kwargs)
    elif name=="PGD":
        return PGD(*args,**kwargs)

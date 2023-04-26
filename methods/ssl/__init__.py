from collections import OrderedDict

from .base import ContinualSSL
from .simclr import SimCLR
from .simsiam import SimSiam
from .barlowtwins import BarlowTwins
from .vicreg import VICReg

# from .scpp import SupConPP


SSL_METHODS = OrderedDict(
    {
        "simclr": SimCLR,
        "simsiam": SimSiam,
        "barlowtwins": BarlowTwins,
        "vicreg": VICReg,
    }
)

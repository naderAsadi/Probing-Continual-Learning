from collections import OrderedDict

from .er import ER
from .er_ace import ER_ACE
from .er_aml import ER_AML_Triplet, ER_AML
from .mir import MIR
from .icarl import ICARL
from .agem import AGEM, AGEMpp
from .cope import CoPE
from .der import DER, DERpp
from .iid import IID, IIDpp
from .moco import MoCo
from .lwf import LwF
from .ewc import EWC
from .ssl import SSL_METHODS

from .repe import RePE
from .supcon_tta import SupConTTA
from .spc import SPC


METHODS = OrderedDict(
    {
        "icarl": ICARL,
        "er": ER,
        "er_ace": ER_ACE,
        "er_aml": ER_AML,
        "er_aml_triplet": ER_AML_Triplet,
        "mir": MIR,
        "iid": IID,
        "iid++": IIDpp,
        "der": DER,
        "der++": DERpp,
        "agem": AGEM,
        "agem++": AGEMpp,
        "moco": MoCo,
        "lwf": LwF,
        "ewc": EWC,
        #'cope'  : cope,
        "repe": RePE,
        "supcontta": SupConTTA,
        "spc": SPC,
    }
)


def get_method(name):
    if name in SSL_METHODS.keys():
        return SSL_METHODS[name]
    return METHODS[name]

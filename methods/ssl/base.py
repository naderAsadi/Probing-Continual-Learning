import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from methods.er import ER

# from model import normalize, LinearClassifier
from models import LinearClassifier


class TwoCropTransform(torch.nn.Module):
    """Create two crops of the same image"""

    def __init__(self, transform):
        super(TwoCropTransform, self).__init__()
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class ContinualSSL(ER):
    def __init__(self, model, logger, train_tf, args):
        super(ContinualSSL, self).__init__(model, logger, train_tf, args)
        self.train_tf = TwoCropTransform(transform=self.train_tf)

    @property
    def name(self):

        return ""

    @property
    def cost(self):
        """return the number of passes (fwd + bwd = 2) through the model for training on one sample"""

        raise NotImplementedError

    def process_inc(self, inc_data):
        """get loss from incoming data"""

        raise NotImplementedError

    def process_re(self, re_data):
        """get loss from rehearsal data"""

        return 0

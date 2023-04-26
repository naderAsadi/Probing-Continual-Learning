import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.ssl import ContinualSSL
from models import normalize


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(ContinualSSL):
    def __init__(self, model, logger, train_tf, args):
        super(BarlowTwins, self).__init__(model, logger, train_tf, args)

        self.lambd = 0.0051
        self.bn = nn.BatchNorm1d(self.args.projection_size, affine=False).to(
            self.device
        )

    @property
    def name(self):
        args = self.args
        return f"BarlowTwins{args.dataset}_U{args.unsupervised}_M{args.mem_size}_Augs{args.use_augs}_E{args.n_epochs}"

    @property
    def cost(self):
        """return the number of passes (fwd + bwd = 2) through the model for training on one sample"""
        return 0  # Hardcoded for now

    def _loss(self, x1, x2):
        z1 = self.model(x1)
        z2 = self.model(x2)

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        # torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss

    def process_inc(self, inc_data):
        """get loss from incoming data"""

        # create other views of input
        x1, x2 = self.train_tf(inc_data["x"])

        loss = self._loss(x1, x2)

        return loss

    def process_re(self, re_data):
        """get loss from rehearsal data"""

        return 0

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.ssl import ContinualSSL
from models import normalize


def invariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes mse loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: invariance loss (mean squared error).
    """

    return F.mse_loss(z1, z2)


def variance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes variance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: variance regularization loss.
    """

    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)
    std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
    return std_loss


def covariance_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Computes covariance loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
    Returns:
        torch.Tensor: covariance regularization loss.
    """

    N, D = z1.size()

    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)
    cov_z1 = (z1.T @ z1) / (N - 1)
    cov_z2 = (z2.T @ z2) / (N - 1)

    diag = torch.eye(D, device=z1.device)
    cov_loss = (
        cov_z1[~diag.bool()].pow_(2).sum() / D + cov_z2[~diag.bool()].pow_(2).sum() / D
    )
    return cov_loss


def vicreg_loss_func(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_loss_weight: float = 25.0,
    var_loss_weight: float = 25.0,
    cov_loss_weight: float = 1.0,
) -> torch.Tensor:
    """Computes VICReg's loss given batch of projected features z1 from view 1 and
    projected features z2 from view 2.
    Args:
        z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
        z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        sim_loss_weight (float): invariance loss weight.
        var_loss_weight (float): variance loss weight.
        cov_loss_weight (float): covariance loss weight.
    Returns:
        torch.Tensor: VICReg loss.
    """

    sim_loss = invariance_loss(z1, z2)
    var_loss = variance_loss(z1, z2)
    cov_loss = covariance_loss(z1, z2)

    loss = (
        sim_loss_weight * sim_loss
        + var_loss_weight * var_loss
        + cov_loss_weight * cov_loss
    )
    return loss


class VICReg(ContinualSSL):
    def __init__(self, model, logger, train_tf, args):
        super(VICReg, self).__init__(model, logger, train_tf, args)

        self.sim_loss_weight = 25
        self.var_loss_weight = 25
        self.cov_loss_weight = 1.0

        self.bn = nn.BatchNorm1d(self.args.projection_size, affine=False).to(
            self.device
        )

    @property
    def name(self):
        args = self.args
        return f"VICReg{args.dataset}_U{args.unsupervised}_M{args.mem_size}_Augs{args.use_augs}_E{args.n_epochs}"

    @property
    def cost(self):
        """return the number of passes (fwd + bwd = 2) through the model for training on one sample"""
        return 0  # Hardcoded for now

    def process_inc(self, inc_data):
        """get loss from incoming data"""

        # create other views of input
        x1, x2 = self.train_tf(inc_data["x"])

        z1, z2 = self.model(x1), self.model(x2)

        loss = vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        return loss

    def process_re(self, re_data):
        """get loss from rehearsal data"""

        return 0

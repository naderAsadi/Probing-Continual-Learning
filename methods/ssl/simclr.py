import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.ssl import ContinualSSL
from models import normalize


class SimCLR(ContinualSSL):
    def __init__(self, model, logger, train_tf, args):
        super(SimCLR, self).__init__(model, logger, train_tf, args)

        self.contrast_mode = "all"
        self.temperature = args.supcon_temperature
        self.base_temperature = 0.07

    @property
    def name(self):
        args = self.args
        return f"SimCLR_{args.dataset}_U{args.unsupervised}_M{args.mem_size}_Augs{args.use_augs}_E{args.n_epochs}"

    @property
    def cost(self):
        """return the number of passes (fwd + bwd = 2) through the model for training on one sample"""
        return 0  # Hardcoded for now

    def simclr_loss(self, features, labels=None):

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown contrast mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(self.device),
            0,
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    def process_inc(self, inc_data):
        """get loss from incoming data"""

        # create other views of input
        x1, x2 = self.train_tf(inc_data["x"])
        data = torch.cat((x1, x2), dim=0)
        bsz = inc_data["x"].shape[0]

        features = self.model(data)
        features = F.normalize(features, dim=1)  # normalize embedding
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # compute loss
        if self.args.unsupervised:
            loss = self.simclr_loss(features, labels=None)
        else:
            loss = self.simclr_loss(features, labels=inc_data["y"])

        return loss

    def process_re(self, re_data):
        """get loss from rehearsal data"""

        return 0

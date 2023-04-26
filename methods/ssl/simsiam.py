import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.ssl import ContinualSSL
from models import PredictionMLP, normalize, FEATURE_DIMS
from utils import set_optimizer


class SimSiam(ContinualSSL):
    def __init__(self, model, logger, train_tf, args):
        super(SimSiam, self).__init__(model, logger, train_tf, args)

        self.prediction_head = PredictionMLP(
            dim_in=self.args.projection_size,
            hidden_dim=self.args.prediction_hidden_size,
            out_dim=self.args.projection_size,
        )
        self.prediction_head = self.prediction_head.to(self.device)

        self.opt = set_optimizer(
            args,
            parameters=list(self.model.parameters())
            + list(self.prediction_head.parameters()),
        )

        self.mode = "simplified"

    @property
    def name(self):
        args = self.args
        return f"SimSiam_{args.dataset}_U{args.unsupervised}_M{args.mem_size}_Augs{args.use_augs}_E{args.n_epochs}"

    @property
    def cost(self):
        """return the number of passes (fwd + bwd = 2) through the model for training on one sample"""
        return 0  # Hardcoded for now

    def simsiam_loss(self, p, z):
        if self.mode == "original":
            z = z.detach()  # stop gradient
            p = F.normalize(p, dim=1)  # l2-normalize
            z = F.normalize(z, dim=1)  # l2-normalize
            return -(p * z).sum(dim=-1).mean()

        elif self.mode == "simplified":
            return -F.cosine_similarity(p, z.detach(), dim=-1).mean()
        else:
            raise Exception

    def process_inc(self, inc_data):
        """get loss from incoming data"""

        # create other views of input
        x1, x2 = self.train_tf(inc_data["x"])
        data = torch.cat((x1, x2), dim=0)
        bsz = inc_data["x"].shape[0]

        features = self.model(data)  # features after projection head

        z1, z2 = torch.split(features, [bsz, bsz], dim=0)
        p1, p2 = self.prediction_head(z1), self.prediction_head(z2)

        loss = 1 + (self.simsiam_loss(p=p1, z=z2) + self.simsiam_loss(p=p2, z=z1)) / 2

        return loss

    def process_re(self, re_data):
        """get loss from rehearsal data"""

        return 0

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from losses import SupConLoss
from methods.ssl import ContinualSSL
from utils import *


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.FloatTensor):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.FloatTensor):
        return F.hardtanh(grad_output)


class SPCHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_dim: int = 256,
        output_dim: int = 8192,
    ):
        super(SPCHead, self).__init__()

        self.bottleneck = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.classifier = nn.Linear(output_dim, n_classes)

    def return_spc(self, x: torch.FloatTensor):
        x = self.bottleneck(x)
        x = STEFunction.apply(x)  # binarize features

        return x

    def classify(self, spc: torch.FloatTensor):
        return self.classifier(spc)

    def forward(self, x: torch.FloatTensor):
        x = self.return_spc(x)  # get binary features
        x = self.classify(x)  # classify binary features

        return x


class SPC(ContinualSSL):
    def __init__(self, model, logger, train_tf, args):
        super(SPC, self).__init__(model, logger, train_tf, args)

        SPC_DIM = 8192

        self.head = SPCHead(
            input_dim=self.model.encoder.last_hid,
            n_classes=self.args.n_classes,
            hidden_dim=256,
            output_dim=SPC_DIM,
        ).to(self.device)

        self.opt = set_optimizer(
            args=self.args,
            parameters=[
                {"params": self.model.parameters()},
                {
                    "params": self.head.parameters(),
                    "lr": 0.01,
                    "momentum": 0.0,
                    "weight_decay": 0.0,
                },
            ],
        )

        self.supcon_loss = SupConLoss(
            temperature=self.args.supcon_temperature, device=self.device
        )

        self.SPCodes = {
            str(l): torch.zeros(SPC_DIM).to(self.device)
            for l in range(self.args.n_classes)
        }

    @property
    def name(self):
        args = self.args
        return f"SPC_{args.dataset}_M{args.mem_size}_Augs{args.use_augs}_TF{args.task_free}_E{args.n_epochs}"

    def train(self):
        self.model.train()
        self.head.train()

    def _spc_classification(self, features: torch.FloatTensor):
        SPCodes = torch.stack(list(self.SPCodes.values()), dim=0)
        spc = self.head.return_spc(features)

        pred = []
        for i in range(spc.size(0)):
            print(torch.sum(spc[i, ...]), SPCodes)
            raise ValueError()
            logits = torch.logical_and(spc[i, ...], SPCodes).sum(dim=1)
            pred.append(torch.zeros(self.args.n_classes))
            print(logits)
            pred[i][torch.argmax(logits)] = 1

        pred = torch.stack(pred, dim=0).to(self.device)

        return pred

    def predict(self, x, task=None):
        logits = None
        features = self.model.return_hidden(x, layer=self.args.eval_layer)

        if self.args.multilinear_eval:
            logits = self.linear_heads[str(task)](features)
        elif self.args.singlelinear_eval:
            if self.args.keep_training_data:
                logits = list(self.linear_heads.values())[0](features)
            else:
                # Do the pred stuff here
                logits = self.head(features)

        return logits, features

    def _pattern_attractor(self, spc: torch.FloatTensor, labels: torch.FloatTensor):

        for l in labels.unique().tolist():
            spc_l = spc[labels == l]
            spccode = STEFunction.apply(torch.sum(spc_l, dim=0))

            if torch.sum(self.SPCodes[str(l)]) == 0:
                self.SPCodes[str(l)] = spccode
            else:
                self.SPCodes[str(l)] = 0.9 * self.SPCodes[str(l)] + 0.1 * spccode

    def _get_supcon_loss(self, features, labels):
        bsz = features.size(0) // 2

        proj_features = self.model.forward_classifier(features)
        proj_features = F.normalize(proj_features, dim=1)  # normalize embedding
        f1, f2 = torch.split(proj_features, [bsz, bsz], dim=0)
        proj_features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        return self.supcon_loss(proj_features, labels=labels)

    def _get_spc_loss(self, features, labels):
        spc = self.head.return_spc(features.detach())
        logits = self.head.classify(spc)

        # calculate loss
        ce_loss = F.cross_entropy(logits, labels)
        reg_loss = torch.norm(spc, 1)

        # aggregate SPCodes for evaluation
        self._pattern_attractor(spc, labels)

        return ce_loss, reg_loss

    def process_inc(self, inc_data):
        """get loss from incoming data"""

        x1, x2 = self.train_tf(inc_data["x"])
        data = torch.cat((x1, x2), dim=0)
        bsz = inc_data["x"].shape[0]
        features = self.model.return_hidden(data)

        supcon_loss = self._get_supcon_loss(features, labels=inc_data["y"])
        spc_ce_loss, spc_reg_loss = self._get_spc_loss(
            features, labels=inc_data["y"].repeat(2)
        )

        # print(supcon_loss.item(), spc_ce_loss.item(), spc_reg_loss.item())

        return supcon_loss + spc_ce_loss  # + 0.5 * spc_reg_loss

from typing import Any, Callable, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from losses import SupConLoss
from methods.er import ER
from utils import *


class Prototypes(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        n_classes_per_task: int,
        n_tasks: int,
        half_iid: bool = False,
    ):
        super(Prototypes, self).__init__()

        self.heads = self._create_prototypes(
            dim_in=feat_dim,
            n_classes=n_classes_per_task,
            n_heads=n_tasks,
            half_iid=half_iid,
        )

    def _create_prototypes(
        self, dim_in: int, n_classes: int, n_heads: int, half_iid: bool = False
    ) -> torch.nn.ModuleDict:

        first_head_id = 0
        if half_iid:
            first_head_id = (n_heads // 2) - 1
            first_head_n_classes = n_classes * (n_heads // 2)

        layers = {}
        for t in range(first_head_id, n_heads):

            if half_iid and (t == first_head_id):
                layers[str(t)] = nn.Linear(dim_in, first_head_n_classes, bias=False)
            else:
                layers[str(t)] = nn.Linear(dim_in, n_classes, bias=False)

        return nn.ModuleDict(layers)

    def forward(self, x: torch.FloatTensor, task_id: int) -> torch.FloatTensor:
        out = self.heads[str(task_id)](x)
        return out


class RePE(ER):
    def __init__(
        self, model: nn.Module, logger: Logger, train_tf: Callable, args: Dict[str, Any]
    ) -> "RePE":
        super(RePE, self).__init__(model, logger, train_tf, args)

        self.prototypes = Prototypes(
            feat_dim=self.model.encoder.last_hid,
            n_classes_per_task=self.args.n_classes_per_task,
            n_tasks=self.args.n_tasks,
            half_iid=args.half_iid,
        ).to(self.device)

        self.opt = set_optimizer(
            args=self.args,
            parameters=[
                {"params": self.model.parameters()},
                {
                    "params": self.prototypes.parameters(),
                    "lr": self.args.prototypes_lr,
                    "momentum": 0.0,
                    "weight_decay": 0.0,
                },
            ],
        )

        self.supcon_loss = SupConLoss(
            temperature=self.args.supcon_temperature, device=self.device
        )

        self.first_task_id = 0
        if self.args.half_iid:
            self.first_task_id = (self.args.n_tasks // 2) - 1

        self.distill_temp = self.args.distill_temp
        self.prev_model = None
        self.prev_prototypes = None

    @property
    def name(self) -> str:
        args = self.args
        return f"RePE_{args.dataset}_M{args.mem_size}_Augs{args.use_augs}_TF{args.task_free}_E{args.n_epochs}"

    @property
    def cost(self) -> float:
        return (
            3
            * (self.args.batch_size + self.args.buffer_batch_size)
            / self.args.batch_size
        )

    def train(self) -> None:
        self.model.train()
        self.prototypes.train()

    def _record_state(self) -> None:
        self.prev_model = copy.deepcopy(self.model)
        self.prev_prototypes = copy.deepcopy(self.prototypes)
        # self.prev_model.eval()
        # self.prev_prototypes.eval()

    def _distillation_loss(
        self, current_out: torch.FloatTensor, prev_out: torch.FloatTensor
    ) -> torch.FloatTensor:

        log_p = torch.log_softmax(current_out / self.distill_temp, dim=1)  # student
        q = torch.softmax(prev_out / self.distill_temp, dim=1)  # teacher
        # result = torch.nn.KLDivLoss(reduction="batchmean")(log_p, q)
        result = torch.sum(-q * log_p, dim=-1).mean()

        return result

    def relation_distillation_loss(
        self, features: torch.FloatTensor, data: torch.FloatTensor, current_task_id: int
    ) -> torch.FloatTensor:

        if self.prev_model is None:
            return 0.0

        old_model_preds = dict()
        new_model_preds = dict()

        with torch.inference_mode():
            old_features = self.prev_model.return_hidden(data)

        for task_id in range(self.first_task_id, current_task_id):
            with torch.inference_mode():
                old_model_preds[task_id] = self._get_scores(
                    old_features, prototypes=self.prev_prototypes, task_id=task_id
                )
            new_model_preds[task_id] = self._get_scores(
                features, prototypes=self.prototypes, task_id=task_id
            )

        dist_loss = 0
        for task_id in old_model_preds.keys():
            dist_loss += self._distillation_loss(
                current_out=new_model_preds[task_id],
                prev_out=old_model_preds[task_id].clone(),
            )

        return dist_loss

    def _classify(self, features: torch.FloatTensor, task_id: int) -> torch.FloatTensor:
        # Copy previous weights
        no_normed_weights = self.prototypes.heads[str(task_id)].weight.data.clone()
        # Normalize weights and features
        self.prototypes.heads[str(task_id)].weight.copy_(
            F.normalize(self.prototypes.heads[str(task_id)].weight.data, dim=1, p=2)
        )
        features = F.normalize(features, dim=1, p=2)  # pass through projection head

        output = self.prototypes(features, task_id=task_id)
        self.prototypes.heads[str(task_id)].weight.copy_(no_normed_weights)

        return output

    def _classify_CI(self, features: torch.FloatTensor) -> torch.FloatTensor:

        heads = list(self.prototypes.heads.values())

        prototypes = heads[0].weight.data.clone()
        for head in heads[1:]:
            prototypes = torch.cat([prototypes, head.weight.data.clone()], dim=0)

        prototypes = F.normalize(prototypes, dim=1, p=2)
        features = F.normalize(features, dim=1, p=2)  # pass through projection head

        output = F.linear(input=features, weight=prototypes)

        return output

    # def _classify_CI(self, features: torch.FloatTensor) -> torch.FloatTensor:

    #     heads = list(self.prototypes.heads.values())

    #     outputs = []
    #     for k, h in self.prototypes.heads.items():
    #         # prototypes = torch.cat([prototypes, head.weight.data.clone()], dim=0)
    #         head = copy.deepcopy(h)
    #         head.weight.copy_(F.normalize(head.weight.data, dim=1, p=2))

    #         features = F.normalize(features, dim=1, p=2)
    #         outputs.append(head(features))

    #     outputs = torch.cat(outputs, dim=-1)

    #     return outputs

    def _get_scores(
        self, features: torch.FloatTensor, prototypes: Prototypes, task_id: int
    ) -> torch.FloatTensor:

        nobout = F.linear(features, prototypes.heads[str(task_id)].weight)
        wnorm = torch.norm(prototypes.heads[str(task_id)].weight, dim=1, p=2)
        nobout = nobout / wnorm
        return nobout

    def linear_loss(
        self,
        features: torch.FloatTensor,
        labels: torch.Tensor,
        current_task_id: int,
        lam: int = 1,
    ) -> torch.FloatTensor:

        if lam == 0:
            features = features.detach().clone()  # [0:labels.size(0)]

        nobout = F.linear(features, self.prototypes.heads[str(current_task_id)].weight)
        wnorm = torch.norm(
            self.prototypes.heads[str(current_task_id)].weight, dim=1, p=2
        )
        nobout = nobout / wnorm
        feat_norm = torch.norm(features, dim=1, p=2)

        if not current_task_id == self.first_task_id:
            labels -= current_task_id * self.args.n_classes_per_task  # shift targets
        indecies = labels.unsqueeze(1)
        out = nobout.gather(1, indecies).squeeze()
        out = out / feat_norm
        loss = sum(1 - out) / out.size(0)

        return loss

    def _prototypes_contrast_loss(self, task_id: int):
        # anchor = self.prototypes.heads[str(task_id)].weight.
        contrast_prot = []
        for key, head in self.prototypes.heads.items():
            if int(key) < task_id:
                contrast_prot.append(copy.deepcopy(head).weight.data)

        if len(contrast_prot) == 0:
            return 0.0

        contrast_prot = F.normalize(torch.cat(contrast_prot, dim=-1), dim=1, p=2)
        anchors = F.normalize(
            self.prototypes.heads[str(task_id)].weight.data, dim=1, p=2
        )

        logits = torch.div(
            torch.matmul(anchors.T, contrast_prot), self.args.supcon_temperature
        )
        log_prob = torch.log(torch.exp(logits).sum(1))
        loss = -log_prob.sum() / log_prob.size(0)

        return loss

    def predict(self, x: torch.FloatTensor, task_id: int = None) -> torch.FloatTensor:
        """used for eval time prediction"""
        logits = None
        features = self.model.return_hidden(x, layer=self.args.eval_layer)

        if self.args.multilinear_eval:
            logits = self.linear_heads[str(task_id)](features)
        elif self.args.singlelinear_eval:
            if self.args.keep_training_data:
                logits = list(self.linear_heads.values())[0](features)
            else:
                if self.args.task_incremental:
                    logits = self._classify(features, task_id)
                else:
                    logits = self._classify_CI(features)

        return logits, features

    def process_inc(self, inc_data: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        """get loss from incoming data"""

        x1, x2 = self.train_tf(inc_data["x"]), self.train_tf(inc_data["x"])
        aug_data = torch.cat((x1, x2), dim=0)
        bsz = inc_data["x"].shape[0]

        features = self.model.return_hidden(aug_data)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)

        # SupCon Loss
        proj_features = self.model.forward_classifier(features)
        proj_features = F.normalize(proj_features, dim=1)  # normalize embedding
        proj_f1, proj_f2 = torch.split(proj_features, [bsz, bsz], dim=0)
        proj_features = torch.cat([proj_f1.unsqueeze(1), proj_f2.unsqueeze(1)], dim=1)
        supcon_loss = self.supcon_loss(proj_features, labels=inc_data["y"])

        # Distillation loss
        loss_d = self.relation_distillation_loss(
            features, data=aug_data, current_task_id=inc_data["t"]
        )

        # Prorotypes loss
        loss_p = self.linear_loss(
            features.detach().clone(),
            labels=inc_data["y"].repeat(2),
            current_task_id=inc_data["t"],
        )

        # Prototypes contrast loss
        # loss_c = self._prototypes_contrast_loss(task_id=inc_data["t"])

        return (
            supcon_loss
            + self.args.prototypes_coef * loss_p
            + self.args.distill_coef * loss_d
            # + loss_c
        )

    def on_task_finish(self, task_id: int):
        super().on_task_finish(task_id)

        self._record_state()

    def eval_agent(self, loader, task, mode="valid"):

        self.prototypes.eval()
        accs = super().eval_agent(loader, task, mode)
        return accs

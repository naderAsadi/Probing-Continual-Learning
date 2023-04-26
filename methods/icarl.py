import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from copy import deepcopy

from methods.er import ER


class ICARL(ER):
    def __init__(self, model, logger, train_tf, args):
        super(ICARL, self).__init__(model, logger, train_tf, args)

        assert not args.task_free

        self.D_C = 1

        self._centroids = None
        self._old_model = None

        self.bce_sum = nn.BCEWithLogitsLoss(reduction="sum")

    @property
    def name(self):
        args = self.args
        return f"ICARL_{args.dataset}_M{args.mem_size}_Augs{args.use_augs}_TF{args.task_free}_DC{args.distill_coef}"

    @property
    def cost(self):
        cost = (
            3
            * (self.args.batch_size + self.args.buffer_batch_size)
            / self.args.batch_size
        )

        # the extra fwd pass for the distillation step
        cost += 1 * (self.args.buffer_batch_size) / self.args.batch_size

        return cost

    def _on_task_switch(self):
        self._old_model = deepcopy(self.model)
        self._old_model.eval()

    @torch.no_grad()
    def _calculate_centroids(self):
        print("calculating centroids")
        buffer = self.buffer

        n_batches = len(buffer) // 512 + 1

        hid_size = self.model.return_hidden(buffer.bx[:2]).size(-1)

        arr_D = torch.arange(hid_size).to(buffer.bx.device)

        protos = buffer.bx.new_zeros(size=(self.args.n_classes, hid_size))
        count = buffer.by.new_zeros(size=(self.args.n_classes,))

        for i in range(n_batches):
            idx = range(i * 512, min(len(buffer), (i + 1) * 512))
            xx, yy = buffer.bx[idx], buffer.by[idx]

            hid_x = self.model.return_hidden(xx)

            b_proto = torch.zeros_like(protos)
            b_count = torch.zeros_like(count)

            b_count.scatter_add_(0, yy, torch.ones_like(yy))

            out_idx = arr_D.view(1, -1) + yy.view(-1, 1) * hid_size
            b_proto = (
                b_proto.view(-1)
                .scatter_add(0, out_idx.view(-1), hid_x.view(-1))
                .view_as(b_proto)
            )

            protos += b_proto
            count += b_count

        self._centroids = protos / count.view(-1, 1)

        # mask out unobserved centroids
        self._centroids[count < 1] = -1e9

    def process_inc(self, inc_data):
        """get a loss signal from data"""

        # build label
        aug_data = self.train_tf(inc_data["x"])

        logits = self.model(aug_data)
        label = F.one_hot(inc_data["y"], num_classes=logits.size(-1)).float()

        loss = self.bce_sum(logits.view(-1), label.view(-1)).sum()
        loss = loss / logits.size(0)
        return loss

    def process_re(self, re_data):
        """get loss from incoming data"""

        aug_data = self.train_tf(re_data["x"])

        loss = 0

        if self._old_model is not None:
            with torch.no_grad():
                tgt = F.sigmoid(self._old_model(aug_data))

            logits = self.model(aug_data)
            loss = self.bce_sum(logits.view(-1), tgt.view(-1)) / logits.size(0)

        return loss

    def observe(self, inc_data):
        if inc_data["t"] != self.task:
            self._on_task_switch()

        super().observe(inc_data)

        # mask centroids as out of sync
        self._centroids = None

    def predict(self, x, task=None):
        if self._centroids is None:
            self._calculate_centroids()

        # calculate distance matrix between incoming and _centroids
        hid_x = self.model.return_hidden(x)  # bs x D
        protos = self._centroids

        dist = (protos.unsqueeze(0) - hid_x.unsqueeze(1)).pow(2).sum(-1)

        return -dist, None

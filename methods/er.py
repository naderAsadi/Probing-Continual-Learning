import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from methods.base import Method
from utils import *


class ER(Method):
    def __init__(self, model, logger, train_tf, args):
        super(ER, self).__init__(model, logger, train_tf, args)

        # note that this is not used for task-free methods
        self.task = torch.LongTensor([0]).to(self.device)

        self.sample_kwargs = {
            "amt": args.buffer_batch_size,
            "exclude_task": None if args.task_free else self.task,
        }

        self.n_ok = 0

    @property
    def name(self):
        args = self.args
        return f"ER_{args.dataset}_M{args.mem_size}_Augs{args.use_augs}_TF{args.task_free}_E{args.n_epochs}"

    @property
    def cost(self):
        return (
            3
            * (self.args.batch_size + self.args.buffer_batch_size)
            / self.args.batch_size
        )

    def predict(self, x, task=None):
        """used for eval time prediction"""
        logits = None
        features = self.model.return_hidden(x, layer=self.args.eval_layer)

        if self.args.multilinear_eval:
            logits = self.linear_heads[str(task)](features)
        elif self.args.singlelinear_eval:
            if self.args.keep_training_data:
                logits = list(self.linear_heads.values())[0](features)
            else:
                logits = self.model.forward_classifier(features, task=task)

        return logits, features

    def _process(self, data):
        """get a loss signal from data"""

        aug_data = self.train_tf(data["x"])
        pred = self.model(aug_data, task=self.task)

        if self.args.task_incremental and not self.args.iid_split:
            data["y"] = data["y"] - data["t"] * self.args.n_classes_per_task

        loss = self.loss(pred, data["y"])

        return loss

    def process_inc(self, inc_data):
        """get loss from incoming data"""

        return self._process(inc_data)

    def process_re(self, re_data):
        """get loss from rehearsal data"""

        if not self.args.task_incremental:
            return self._process(re_data)

        aug_data = self.train_tf(re_data["x"])
        features = self.model.return_hidden(aug_data, layer=self.args.eval_layer)
        loss = 0
        for t in re_data["t"].unique():
            features_t = features[re_data["t"] == t]
            y = re_data["y"][re_data["t"] == t]
            pred = self.model.forward_classifier(features_t, task=t)
            loss += self.loss(pred, y)

        return loss

    def observe(self, inc_data, update_buffer=True):
        """full step of processing and learning from data"""

        # keep track of current task for task-based methods
        self.task.fill_(inc_data["t"])

        for it in range(self.args.n_iters):
            # --- training --- #
            inc_loss = self.process_inc(inc_data)
            # assert inc_data['x'].size(0) == inc_data['y'].size(0), pdb.set_trace()

            re_loss = 0
            if len(self.buffer) > 0:

                # -- rehearsal starts ASAP. No task id is used
                if self.args.task_free or self.task > 0:
                    re_data = self.buffer.sample(**self.sample_kwargs)

                    re_loss = self.process_re(re_data)

            self.update(inc_loss + re_loss)

        # --- buffer overhead --- #
        if update_buffer:
            self.buffer.add(inc_data)

        return inc_loss + re_loss

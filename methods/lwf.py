import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from methods.base import Method
from utils import *


class LwF(Method):
    def __init__(self, model, logger, train_tf, args):
        super(LwF, self).__init__(model, logger, train_tf, args)

        # note that this is not used for task-free methods
        self.task = torch.LongTensor([0]).to(self.device)

        self.temp = 2
        self.lambda_0 = 1
        self.prev_model = None

    @property
    def name(self):
        args = self.args
        return f"LwF_{args.dataset}_M{args.mem_size}_Augs{args.use_augs}_TF{args.task_free}_E{args.n_epochs}"

    @property
    def cost(self):
        return 0  # Hardcoded for now

    def record_state(self):
        self.prev_model = copy.deepcopy(self.model)

    def _distillation_loss(self, current_out, prev_out):
        log_p = torch.log_softmax(current_out / self.temp, dim=1)
        q = torch.softmax(prev_out / self.temp, dim=1)
        result = torch.nn.KLDivLoss(reduction="batchmean")(log_p, q)
        return result

    def lwf_loss(self, features, data, current_model, current_task_id):
        if self.prev_model is None:
            return 0.0
        predictions_old_tasks_old_model = dict()
        predictions_old_tasks_new_model = dict()
        for task_id in range(self.task):
            with torch.inference_mode():
                predictions_old_tasks_old_model[task_id] = self.prev_model(
                    data, task=task_id
                )
            predictions_old_tasks_new_model[task_id] = current_model.forward_classifier(
                features, task=task_id
            )
        dist_loss = 0
        for task_id in predictions_old_tasks_old_model.keys():
            dist_loss += self._distillation_loss(
                current_out=predictions_old_tasks_new_model[task_id],
                prev_out=predictions_old_tasks_old_model[task_id].clone(),
            )
        return self.lambda_0 * dist_loss

    def process_inc(self, features, labels):
        """get loss from incoming data"""

        if self.args.task_incremental:
            labels -= self.task * self.args.n_classes_per_task

        pred = self.model.forward_classifier(features, task=self.task)
        loss = self.loss(pred, labels)
        return loss

    def observe(self, inc_data):
        """full step of processing and learning from data"""

        # keep track of current task for task-based methods
        self.task.fill_(inc_data["t"])

        aug_data = self.train_tf(inc_data["x"])
        features = self.model.return_hidden(aug_data, layer=self.args.eval_layer)

        inc_loss = self.process_inc(features, labels=inc_data["y"])
        lwf_loss = self.lwf_loss(
            features, inc_data["x"], current_model=self.model, current_task_id=self.task
        )

        loss = inc_loss + lwf_loss
        self.update(loss)

        return loss

    def on_task_finish(self, task):
        super().on_task_finish(task)

        self.record_state()

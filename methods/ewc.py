from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from data.base import get_data_and_tfs
from methods.base import Method
from utils import *


def zero_like_params_dict(model: torch.nn.Module):
    """
    Create a list of (name, parameter), where parameter is initialized to zero.
    The list has as many parameters as the model, with the same size.
    :param model: a pytorch model
    """

    return [(k, torch.zeros_like(p).to(p.device)) for k, p in model.named_parameters()]


def copy_params_dict(model: torch.nn.Module, copy_grad=False):
    """
    Create a list of (name, parameter), where parameter is copied from model.
    The list has as many parameters as model, with the same size.
    :param model: a pytorch model
    :param copy_grad: if True returns gradients instead of parameter values
    """

    if copy_grad:
        return [(k, p.grad.data.clone()) for k, p in model.named_parameters()]
    else:
        return [(k, p.data.clone()) for k, p in model.named_parameters()]


class EWC(Method):
    def __init__(self, model, logger, train_tf, args):
        super(EWC, self).__init__(model, logger, train_tf, args)

        # note that this is not used for task-free methods
        self.task = torch.LongTensor([0]).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

        self.saved_parameters = dict()
        self.importance_matrices = dict()

    def _compute_importance(
        self,
        model: nn.Module,
        criterion: nn.CrossEntropyLoss,
        optimizer: Optimizer,
        train_loader: DataLoader,
        batch_size: int,
        current_task_id: str,
    ):
        """
        Compute EWC importance matrix for each parameter
        """
        model.eval()
        importance_matrix = zero_like_params_dict(model=model)

        train_loader.sampler.set_task(current_task_id)
        for data, targets in train_loader:

            data, targets = data.to(self.device), targets.to(self.device)
            targets -= current_task_id * self.args.n_classes_per_task

            optimizer.zero_grad()
            predictions = model(data, task=current_task_id)

            loss = criterion(predictions, targets)
            loss.backward()

            for (net_param_name, net_param_value), (
                imp_param_name,
                imp_param_value,
            ) in zip(model.named_parameters(), importance_matrix):
                assert net_param_name == imp_param_name
                if net_param_value.grad is not None:
                    imp_param_value += net_param_value.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp_param_value in importance_matrix:
            imp_param_value /= float(len(train_loader))

        return importance_matrix

    def record_state(
        self,
        model: nn.Module,
        criterion: nn.CrossEntropyLoss,
        optimizer: Optimizer,
        train_loader: DataLoader,
        batch_size: int,
        current_task_id: str,
    ):
        # to be called at the end of training each task
        importance_matrix = self._compute_importance(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            train_loader=train_loader,
            batch_size=batch_size,
            current_task_id=current_task_id,
        )

        self.importance_matrices[current_task_id] = importance_matrix
        self.saved_parameters[current_task_id] = copy_params_dict(model)

    def ewc_loss(self, model: nn.Module, current_task_id: str):
        if current_task_id == 0:
            return 0
        loss = 0
        for task_id in range(self.args.n_tasks):
            if task_id == current_task_id:
                break
            for (
                (_, current_parameters),
                (_, saved_parameters),
                (_, importance_weight),
            ) in zip(
                model.named_parameters(),
                self.saved_parameters[task_id],
                self.importance_matrices[task_id],
            ):
                loss += (
                    importance_weight * (current_parameters - saved_parameters).pow(2)
                ).sum()

        return self.args.ewc_lambda * loss

    def process_inc(self, data: torch.FloatTensor, labels: torch.FloatTensor):
        """get loss from incoming data"""

        if self.args.task_incremental:
            labels -= self.task * self.args.n_classes_per_task

        pred = self.model(data, task=self.task)
        loss = self.loss(pred, labels)

        return loss

    def observe(self, inc_data: Dict[str, torch.FloatTensor]):
        """full step of processing and learning from data"""

        # keep track of current task for task-based methods
        self.task.fill_(inc_data["t"])

        aug_data = self.train_tf(inc_data["x"])

        inc_loss = self.process_inc(aug_data, labels=inc_data["y"])

        ewc_loss = self.ewc_loss(model=self.model, current_task_id=inc_data["t"])

        loss = inc_loss + ewc_loss
        self.update(loss)

        return loss

    def on_task_finish(self, task: int):
        super().on_task_finish(task)

        _, train_loader, _, _ = get_data_and_tfs(self.args)

        self.record_state(
            model=self.model,
            criterion=self.criterion,
            optimizer=self.opt,
            train_loader=train_loader,
            batch_size=self.args.batch_size,
            current_task_id=task,
        )

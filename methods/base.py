import copy
from os import listdir
from os.path import isfile, join
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import FlopCountAnalysis as FCA

from buffer import Buffer
from models import LinearClassifier, FEATURE_DIMS
from utils import set_optimizer, CudaCKA

import wandb


# Abstract Class
class Method(nn.Module):
    def __init__(self, model, logger, train_tf, args):
        super(Method, self).__init__()

        self.args = args
        self.model = model
        self.train_tf = train_tf
        self.logger = logger

        self.device = next(model.parameters()).device
        self.buffer = Buffer(
            capacity=args.mem_size, input_size=args.input_size, device=self.device
        )

        self.loss = F.cross_entropy
        self.opt = set_optimizer(args, parameters=list(self.model.parameters()))

        self.n_fwd, self.n_bwd = 0, 0

        self.logger.register_name(self.name)

        self.linear_heads = {}  # for linear evaluation
        self.model_history = {}

        if self.args.cka_eval:
            self.cuda_cka = CudaCKA(self.device)

    @property
    def name(self):
        return ""

    @property
    def cost(self):
        """return the number of passes (fwd + bwd = 2) through the model for training on one sample"""

        raise NotImplementedError

    @property
    def one_sample_flop(self):
        if not hasattr(self, "_train_cost"):
            input = torch.FloatTensor(size=(1,) + self.args.input_size).to(self.device)
            flops = FCA(self.model, input)
            self._train_cost = flops.total() / 1e6  # MegaFlops

        return self._train_cost

    def observe(self, inc_data, rehearse=False):
        """full step of processing and learning from data"""

        raise NotImplementedError

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

    def copy_model(self):
        return copy.deepcopy(self.model).eval()

    def save_model(self, model, task, epoch):
        print(f"\nSaving task {task} model")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.opt.state_dict(),
            },
            self.args.snapshot_path + self.name + f"_Task{task}.pt",
        )

    def load_model_history(self):
        snapshots_path = [
            f
            for f in listdir(self.args.snapshot_path)
            if isfile(join(self.args.snapshot_path, f))
        ]
        for path in snapshots_path:
            task_t = path[path.index("Task") + 4 : path.index(".pt")]
            checkpoint = torch.load(self.args.snapshot_path + path)
            self.model_history[task_t] = self.copy_model()
            self.model_history[task_t].load_state_dict(checkpoint["model_state_dict"])

            print(f"Loaded {idx + 1}/{len(snapshots_path)} checkpoints", end="\r")

        print(
            f"{len(self.model_history.keys())} checkpoints loaded for {self.args.n_tasks} tasks"
        )

    def update(self, loss):
        """update parameters from loss"""
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    def train(self):
        self.model.train()

    def eval(self, freeze_linear_heads=True):
        self.model.eval()

        for task in self.linear_heads.keys():  # check if this line works correctly
            if freeze_linear_heads:
                self.linear_heads[task].eval()
            else:
                self.linear_heads[task].train()

    def on_task_start(self, *args):

        pass

    def on_task_finish(self, task):
        if self.args.cka_eval:
            self.model_history[str(task)] = self.copy_model()

        if self.args.save_snapshot:
            self.save_model(model=self.model, task=task, epoch=self.args.n_epochs)

    def train_linear_heads(self, train_loader, task_list, sample_all_seen_tasks):
        """train a linear classifier for each seen task"""
        if sample_all_seen_tasks:
            if self.args.iid_split:
                num_classes = self.args.n_classes
            else:
                num_classes = self.args.n_classes_per_task * (task_list[0] + 1)
        else:
            num_classes = self.args.n_classes_per_task

        self.linear_heads = {
            str(t): LinearClassifier(
                name=self.args.model,
                # feat_dim=FEATURE_DIMS[self.args.model][str(self.args.eval_layer)],
                feat_dim=int(
                    self.model.encoder.last_hid / (2 ** (4 - self.args.eval_layer))
                ),
                num_classes=num_classes,
            ).to(self.device)
            for t in task_list
        }

        self.eval(freeze_linear_heads=False)  # just freeze backbone network

        # Train linear heads for each seen task
        for task_t in task_list:
            # for task_t in [0]:
            train_loader.sampler.set_task(
                task_t, sample_all_seen_tasks=sample_all_seen_tasks
            )
            optim = torch.optim.SGD(
                self.linear_heads[str(task_t)].parameters(),
                lr=self.args.eval_lr,
                momentum=0.9,
                weight_decay=0.0,
            )

            for epoch in range(self.args.eval_n_epochs):
                for _, (x, y) in enumerate(train_loader):
                    x, y = x.to(self.device), y.to(self.device)

                    if self.args.multilinear_eval:
                        y = (
                            y - task_t * self.args.n_classes_per_task
                        )  # scale labels to [0 , ...]

                    with torch.no_grad():
                        features = self.model.return_hidden(
                            x, layer=self.args.eval_layer
                        )
                    logits = self.linear_heads[str(task_t)](features.detach())
                    loss = self.loss(logits, y)

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    print(
                        f"Linear head {task_t} | Epoch: {epoch + 1} / {self.args.eval_n_epochs} - Training loss: {loss}",
                        end="\r",
                    )

    @torch.no_grad()
    def evaluate(self, loader, task, mode="valid"):
        """evaluate agent on seen tasks"""
        self.eval(freeze_linear_heads=True)

        accs = np.zeros(shape=(self.args.n_tasks,))
        cka_scores = np.zeros(shape=(self.args.n_tasks,))

        for task_t in range(task + 1):
            # for task_t in range(1):

            n_ok, n_total = 0, 0
            ckas = []
            if not self.args.iid_split:
                loader.sampler.set_task(task_t)

            # iterate over samples from task
            for i, (data, target) in enumerate(loader):

                data, target = data.to(self.device), target.to(self.device)

                if not self.args.iid_split:
                    if (
                        self.args.multilinear_eval or self.args.task_incremental
                    ) and not (
                        self.args.keep_training_data and self.args.singlelinear_eval
                    ):
                        target = (
                            target - task_t * self.args.n_classes_per_task
                        )  # scale labels to [0 , ...]

                logits, features = self.predict(data, task_t)

                n_total += data.size(0)
                if logits is not None:
                    pred = logits.max(1)[1]
                    n_ok += pred.eq(target).sum().item()

                if self.args.cka_eval and task > 0 and task_t < task:
                    old_features = self.model_history[str(task_t)].return_hidden(
                        data, layer=self.args.eval_layer
                    )
                    ckas.append(self.cuda_cka.linear_CKA(features, old_features).item())

            accs[task_t] = (n_ok / n_total) * 100
            cka_scores[task_t] = np.mean(ckas) if len(ckas) > 0 else 0

        avg_acc = np.mean(accs[: task + 1])
        print("\n", "\t".join([str(int(x)) for x in accs]), f"\tAvg Acc: {avg_acc:.2f}")

        # >>>>> Logging <<<<<
        logs = {}

        # Log accuracies
        if avg_acc > 0:
            logs.update(
                {
                    f"{mode}/anytime_last_acc": accs[task],
                    f"{mode}/anytime_acc_avg_seen": avg_acc,
                    f"{mode}/anytime_acc_avg_all": np.mean(accs),
                }
            )
            for task_t, acc in enumerate(accs):
                logs[f"task_accs/task{task_t}"] = acc

        # Log CKA scores
        if self.args.cka_eval:
            # cka_score_task = round(cka_scores[task - 1], 4) if task > 0 else 0
            print(
                "CKA Scores:\n", "\t".join([str(round(x, 4)) for x in cka_scores[:-1]])
            )
            # logs['analysis/cka'] = cka_score_task
            for task_t, cka_score in enumerate(cka_scores[:-1]):
                logs[f"task_cka_scores/task{task_t}"] = cka_score

        self.logger.log_scalars(logs)

        return accs.tolist()

    def eval_agent(self, loader, task, mode="valid"):

        sample_all_seen_tasks = False

        if self.args.multilinear_eval:
            print(f"> Multi-layer Linear Evaluation at Layer {self.args.eval_layer}")
            task_list = range(task + 1)
        elif self.args.singlelinear_eval:
            print(f"> Single-layer Linear Evaluation at Layer {self.args.eval_layer}")
            task_list = [task]
        else:
            print(f"> CKA analysis at Layer {self.args.eval_layer}")

        if self.args.keep_training_data:
            train_loader, eval_loader = loader
            if self.args.singlelinear_eval:
                sample_all_seen_tasks = True
            self.train_linear_heads(train_loader, task_list, sample_all_seen_tasks)
        else:
            eval_loader = loader

        accs = self.evaluate(eval_loader, task, mode)
        return accs

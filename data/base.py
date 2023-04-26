import os
import sys
import pdb
import torch
import numpy as np

from copy import deepcopy
from torchvision import datasets, transforms
from data import *

# ---  Samplers --- #
class ContinualSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, n_tasks, args, smooth=False):
        self.ds = dataset
        self.n_tasks = n_tasks
        self.smooth = smooth

        # print(dataset.data, dataset.targets)
        if isinstance(dataset, torch.utils.data.Subset):
            ds_targets = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            ds_targets = dataset.targets

        self.classes = np.unique(ds_targets)

        ds_targets = np.array(ds_targets)
        self.n_samples = ds_targets.shape[0]
        self.n_classes = self.classes.shape[0]

        assert self.n_classes % n_tasks == 0
        self.cpt = self.n_classes // n_tasks

        self.sample_all_seen_tasks = False

        self.task = None
        self.target_indices = {}

        # for smooth datasets
        self.t = 0
        self.per_class_samples_left = torch.zeros(self.classes.shape[0]).int()

        for label in self.classes:
            self.target_indices[label] = np.squeeze(np.argwhere(ds_targets == label))
            np.random.shuffle(self.target_indices[label])
            self.per_class_samples_left[label] = self.target_indices[label].shape[0]

    def _fetch_task_samples(self, task):
        if self.sample_all_seen_tasks:
            task_classes = self.classes[: self.cpt * (task + 1)]
        else:
            task_classes = self.classes[self.cpt * task : self.cpt * (task + 1)]

        task_samples = []

        for task in task_classes:
            t_indices = self.target_indices[task]
            task_samples += [t_indices]

        task_samples = np.concatenate(task_samples)
        np.random.shuffle(task_samples)

        self.task_samples = task_samples

    def __iter__(self):
        if self.smooth:
            samples_per_class = self.n_samples // self.n_classes
            samples_per_task = self.n_samples // self.n_tasks
            cls = torch.arange(self.n_classes)
            var = (samples_per_class // 2) * 2.5  # 4
            std = np.sqrt(var)
            mu = (2 * cls - 1) * samples_per_class / 2.0

            for t in range(samples_per_task):
                # sample class probs
                pmf = torch.exp((self.t - mu).div_(var)).pow(2).div(-2).div(std * 1.414)
                pmf[self.per_class_samples_left <= 0] = 0
                prob = pmf / pmf.sum()
                class_id = (
                    torch.where((np.random.uniform(0, 1) + prob.cumsum(0)) > 1.0)[0]
                    .min()
                    .item()
                )

                # weird bug with this line of code
                # torch.multinomial(prob, 1).item()

                self.per_class_samples_left[class_id] -= 1
                yield self.target_indices[class_id][
                    self.per_class_samples_left[class_id]
                ]

                import pdb

                # assert (self.per_class_samples_left < 0).sum().item() == 0, pdb.set_trace()

                self.t += 1

        else:
            self._fetch_task_samples(self.task)

            for item in self.task_samples:
                yield item

    def __len__(self):
        samples_per_task = self.n_samples // self.n_tasks
        return samples_per_task

    def set_task(self, task, sample_all_seen_tasks=False):
        self.task = task
        self.sample_all_seen_tasks = sample_all_seen_tasks


class IIDSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, n_tasks, args):

        self.ds = dataset
        self.n_tasks = n_tasks

        if isinstance(dataset, torch.utils.data.Subset):
            ds_targets = np.array(dataset.dataset.targets)[dataset.indices]
        else:
            ds_targets = dataset.targets
        self.classes = np.unique(ds_targets)

        self.n_samples = ds_targets.shape[0]
        self.n_classes = self.classes.shape[0]

        self.sample_all_seen_tasks = False
        self.cpt = self.n_classes

        self.target_indices = {}
        for label in self.classes:
            label_idx = np.squeeze(np.argwhere(dataset.targets == label))
            split_idx = int(label_idx.shape[0] * (1 / n_tasks))

            for t in range(n_tasks):
                if str(t) not in self.target_indices.keys():
                    self.target_indices[str(t)] = [
                        label_idx[t * split_idx : (t + 1) * split_idx]
                    ]
                else:
                    self.target_indices[str(t)].append(
                        label_idx[t * split_idx : (t + 1) * split_idx]
                    )

        for key in self.target_indices.keys():
            np.random.shuffle(self.target_indices[key])

    def _fetch_task_samples(self, task):

        task_samples = []

        if self.sample_all_seen_tasks:
            for t in range(self.n_tasks):
                t_indices = np.concatenate(self.target_indices[str(t)])
                task_samples.append(t_indices)
        else:
            t_indices = np.concatenate(self.target_indices[str(task)])
            task_samples.append(t_indices)

        task_samples = np.concatenate(task_samples)
        np.random.shuffle(task_samples)

        # print(f"samples size: {task_samples.shape}")

        self.task_samples = task_samples

    def __iter__(self):
        self._fetch_task_samples(self.task)

        for item in self.task_samples:
            yield item

    def __len__(self):
        samples_per_task = self.n_samples // self.n_tasks
        return samples_per_task

    def set_task(self, task, sample_all_seen_tasks=False):
        self.task = task
        self.sample_all_seen_tasks = sample_all_seen_tasks


def make_val_from_train(dataset, split=0.9):
    train_ds, val_ds = deepcopy(dataset), deepcopy(dataset)

    train_idx, val_idx = [], []
    for label in np.unique(dataset.targets):
        label_idx = np.squeeze(np.argwhere(dataset.targets == label))
        split_idx = int(label_idx.shape[0] * split)
        train_idx += [label_idx[:split_idx]]
        val_idx += [label_idx[split_idx:]]

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)

    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)

    return train_ds, val_ds


def get_data_and_tfs(args):
    dataset = {
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
        "miniimagenet": MiniImagenet,
        "imagenet32": ImageNet32,
        "imagenet64": ImageNet64,
    }[args.dataset]

    if args.n_tasks == -1:
        args.n_tasks = dataset.default_n_tasks

    H = dataset.default_size
    args.input_size = (3, H, H)

    base_tf = dataset.base_transforms()
    train_tf = dataset.train_transforms(use_augs=args.use_augs)
    eval_tf = dataset.eval_transforms()

    ds_kwargs = {"root": args.data_root, "download": args.download}

    val_ds = test_ds = None

    if args.validation:
        trainval_ds = dataset(train=True, **ds_kwargs)
        train_ds, val_ds = make_val_from_train(trainval_ds)
        train_ds.dataset.transform = base_tf
        val_ds.dataset.transform = eval_tf
    else:
        train_ds = dataset(train=True, transform=base_tf, **ds_kwargs)
        test_ds = dataset(train=False, transform=eval_tf, **ds_kwargs)

    if args.iid_split:
        train_sampler = IIDSampler(train_ds, args.n_tasks, args)
    else:
        train_sampler = ContinualSampler(
            train_ds, args.n_tasks, args, smooth=args.smooth
        )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        num_workers=args.num_workers,
        sampler=train_sampler,
        batch_size=args.batch_size,
    )

    if val_ds is not None:
        val_sampler = ContinualSampler(val_ds, args.n_tasks, args)
        test_loader = None
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            sampler=val_sampler,
        )

    elif test_ds is not None:
        if args.iid_split:
            test_sampler = None
        else:
            test_sampler = ContinualSampler(test_ds, args.n_tasks, args)
        val_loader = None
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            sampler=test_sampler,
        )

    args.n_classes = train_sampler.n_classes
    args.n_classes_per_task = args.n_classes // args.n_tasks
    if args.iid_split:
        args.n_classes_per_task = args.n_classes

    return train_tf, train_loader, val_loader, test_loader

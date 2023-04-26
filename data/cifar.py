import os
import sys
import pdb
import torch
import torch.nn as nn

import kornia
import numpy as np

from copy import deepcopy
from torchvision import datasets, transforms


class CIFAR:

    default_size = 32
    MEAN = (0.4914, 0.4822, 0.4465)
    STD = (0.2023, 0.1994, 0.2010)

    def base_transforms():
        return None

    def train_transforms(use_augs=False):

        H = CIFAR.default_size

        if use_augs:
            tfs = nn.Sequential(
                kornia.augmentation.RandomCrop(size=(H, H), padding=4, fill=-1),
                kornia.augmentation.RandomHorizontalFlip(),
                kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, p=0.8),
                kornia.augmentation.RandomGrayscale(p=0.2),
                # kornia.augmentation.Normalize(CIFAR.MEAN, CIFAR.STD)
            )
        else:
            tfs = nn.Identity()

        return tfs

    def eval_transforms():
        return None


class CIFAR10(CIFAR, datasets.CIFAR10):
    default_n_tasks = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = torch.from_numpy(self.data).float().permute(0, 3, 1, 2)
        self.targets = np.array(self.targets)

        self.data = (self.data / 255.0 - 0.5) * 2.0

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        return x.squeeze(), y


class CIFAR100(CIFAR, datasets.CIFAR100):
    default_n_tasks = 20

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = torch.from_numpy(self.data).float().permute(0, 3, 1, 2)
        self.targets = np.array(self.targets)

        self.data = (self.data / 255.0 - 0.5) * 2.0

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        return x.squeeze(), y

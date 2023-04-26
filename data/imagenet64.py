from typing import Callable, Optional
from pathlib import Path

import kornia
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset


class ImageNet64(Dataset):

    default_size = 64
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        train: Optional[bool] = True,
        **kwargs
    ):
        super(ImageNet64, self).__init__()

        if train:
            path = Path(root).joinpath("imagenet64/train")
        else:
            path = Path(root).joinpath("imagenet64/val")

        data_paths = [*path.glob("*")]

        data = []
        targets = []
        for path in data_paths:
            batch = np.load(path, allow_pickle=True)
            data.append(batch["data"])
            targets.extend(batch["labels"])

        data = np.concatenate(data, axis=0)
        self.data = data.reshape(
            data.shape[0], 3, ImageNet64.default_size, ImageNet64.default_size
        )

        self.targets = [(label - 1) for label in targets]
        self.transform = transform

    @staticmethod
    def base_transforms():
        """base transformations applied to *train* images"""

        return transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def train_transforms(use_augs: Optional[int] = False):
        """extra augs applied over *training* images"""

        H = ImageNet64.default_size
        if use_augs:
            tfs = torch.nn.Sequential(
                kornia.augmentation.RandomCrop(size=(H, H), padding=4, fill=-1),
                kornia.augmentation.RandomHorizontalFlip(p=0.5),
                kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                kornia.augmentation.RandomGrayscale(p=0.2),
                kornia.augmentation.Normalize(ImageNet64.MEAN, ImageNet64.STD),
            )
        else:
            tfs = torch.nn.Sequential(
                kornia.augmentation.Normalize(ImageNet64.MEAN, ImageNet64.STD),
            )

        return tfs

    @staticmethod
    def eval_transforms():
        """base transformations applied during evaluation"""

        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=ImageNet64.MEAN, std=ImageNet64.STD),
            ]
        )

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        x = x.astype(np.single)
        if self.transform is not None:
            x = self.transform(x)

        return x, y

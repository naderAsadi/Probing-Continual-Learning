import os
import kornia
from models.resnet import normalize
import torch
from torchvision import datasets, transforms

from typing import Any, Optional, Tuple


class ImageNet32(datasets.ImageFolder):

    default_size = 32
    default_n_tasks = 100

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(
        self, root, transform, train=True, target_transform=None, download=None
    ) -> None:
        sub_folder = "train" if train else "val"
        root = os.path.join(root, f"imagenet32/{sub_folder}")

        super(ImageNet32, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.transform = transform

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    @staticmethod
    def base_transforms():
        """base transformations applied to *train* images"""

        return transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def train_transforms(use_augs=False):
        """extra augs applied over *training* images"""

        H = ImageNet32.default_size

        if use_augs:
            tfs = torch.nn.Sequential(
                kornia.augmentation.RandomCrop(size=(H, H), padding=4, fill=-1),
                kornia.augmentation.RandomHorizontalFlip(p=0.5),
                kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                kornia.augmentation.RandomGrayscale(p=0.2),
                kornia.augmentation.Normalize(ImageNet32.MEAN, ImageNet32.STD),
            )
        else:
            tfs = torch.nn.Sequential(
                kornia.augmentation.Normalize(ImageNet32.MEAN, ImageNet32.STD),
            )

        return tfs

    @staticmethod
    def eval_transforms():
        """base transformations applied during evaluation"""

        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=ImageNet32.MEAN, std=ImageNet32.STD),
            ]
        )

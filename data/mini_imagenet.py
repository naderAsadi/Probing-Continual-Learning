import os
import pickle as pkl
import kornia
import numpy as np
import torch
from torchvision import datasets, transforms


class MiniImagenet(datasets.VisionDataset):

    default_size = 64
    default_n_tasks = 20

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, root, train=True, transform=None, download=False):

        all_data = []
        for split in ["train", "val", "test"]:
            afile = open(
                os.path.join(root, f"mini_imagenet/mini-imagenet-cache-{split}.pkl"),
                "rb",
            )
            data = pkl.load(afile)["image_data"].reshape(-1, 600, 84, 84, 3)
            all_data += [data]

        all_data = np.concatenate(all_data)

        split = int(1 / 6 * all_data.shape[1])
        all_data = all_data[:, split:] if train else all_data[:, :split]
        all_data = (torch.from_numpy(all_data).float() / 255.0 - 0.5) * 2

        self.data = all_data.reshape(-1, *all_data.shape[2:]).permute(0, 3, 2, 1)
        self.targets = (
            np.arange(100).reshape(-1, 1).repeat(all_data.size(1), axis=1).reshape(-1)
        )
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def base_transforms():
        """base transformations applied to *train* images"""

        return None

    def train_transforms(use_augs=False):
        """extra augs applied over *training* images"""

        H = MiniImagenet.default_size

        if use_augs:
            tfs = torch.nn.Sequential(
                kornia.augmentation.RandomCrop(size=(H, H), padding=4, fill=-1),
                kornia.augmentation.RandomHorizontalFlip(),
                kornia.augmentation.ColorJitter(0.4, 0.4, 0.4, p=0.8),
                kornia.augmentation.RandomGrayscale(p=0.2),
                # kornia.augmentation.Normalize(MiniImagenet.MEAN, MiniImagenet.STD)
            )
        else:
            tfs = torch.nn.Identity()

        return tfs

    def eval_transforms():
        """base transformations applied during evaluation"""

        return None

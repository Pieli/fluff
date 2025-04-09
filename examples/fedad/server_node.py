import torch
import lightning as pl

from torch.utils import data

import sys

sys.path.append("../..")

from fluff import Node
from fluff.datasets import CIFAR10Dataset, MNISTDataset, FMNISTDataset
from fluff.datasets.partitions import BalancedFraction
from fluff.models import CNN

from resnet import ResNet_cifar, ResNet_mnist


class ServerNode(Node):
    def __init__(
        self,
        name: str,
        experiement_name: str,
        model: pl.LightningModule,
        dataset,
        num_workers: int = 2,
        seed=None,
        hp=None,
    ) -> None:
        super().__init__(name, experiement_name, model, dataset, num_workers, seed, hp)

    def setup(self):
        cif_10 = CIFAR10Dataset(
            BalancedFraction(percent=0.9), batch_size=self._dataset.get_batch_size()
        )

        training_dataset = data.Subset(
            self._dataset.train_set,
            self._dataset.train_indices_map,
        )

        train_set = training_dataset
        _, valid_set = data.random_split(
            cif_10.train_set, (0.8, 0.2), generator=torch.Generator().manual_seed(42)
        )
        test_set = cif_10.test_set

        generator = torch.Generator().manual_seed(self._seed) if self._seed else None
        self.train_loader = data.DataLoader(
            train_set,
            batch_size=self._dataset.get_batch_size(),
            shuffle=True,
            num_workers=self._num_workers,
            generator=generator,
        )

        self.val_loader = data.DataLoader(
            valid_set,
            batch_size=self._dataset.get_batch_size(),
            shuffle=False,
            num_workers=self._num_workers,
        )

        self.test_loader = data.DataLoader(
            test_set,
            batch_size=self._dataset.get_batch_size(),
            shuffle=False,
            num_workers=self._num_workers,
        )
        return self


def lam_cnn():
    return CNN(num_classes=10)


def lam_mnist():
    return ResNet_mnist(
        resnet_size=20,
        group_norm_num_groups=2,
        freeze_bn=True,
        freeze_bn_affine=True,
    ).train(True)


def fact(set_name: str):
    match set_name:
        case "cifar10":
            return CIFAR10Dataset, lam_resnet
        case "mnist":
            return MNISTDataset, lam_mnist
        case "fmnist":
            return FMNISTDataset, lam_mnist


def lam_resnet():
    return ResNet_cifar(
        resnet_size=20,
        group_norm_num_groups=2,
        freeze_bn=False,
        freeze_bn_affine=False,
    ).train(True)

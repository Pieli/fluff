from argparse import Namespace

from typing import Tuple, Iterable, List

from models import LitCNN

import torch
from torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import lightning as pl
from torch.utils.data import DataLoader

from aggregator import Noop


class Node:
    def __init__(self, name: str, model: pl.LightningModule) -> None:
        self._name = name
        self.model = model

    def setup(self):
        transform = transforms.ToTensor()

        training_dataset = datasets.CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transform
        )

        train_set_size = int(len(training_dataset) * 0.8)
        valid_set_size = len(training_dataset) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)

        train_set, valid_set = data.random_split(
            training_dataset, [train_set_size, valid_set_size], generator=seed)

        self.train_loader = data.DataLoader(
            train_set, batch_size=32, shuffle=True, num_workers=11)

        self.val_loader = data.DataLoader(
            valid_set, batch_size=32, shuffle=False, num_workers=11)

        self.test_loader = data.DataLoader(
            training_dataset, batch_size=32, shuffle=False, num_workers=11)

        return self

    def train(self, epochs: int) -> None:
        trainer = pl.Trainer(max_epochs=epochs, deterministic=True)

        trainer.fit(model=self.model,
                    train_dataloaders=self.train_loader,
                    val_dataloaders=self.val_loader)

        trainer.test(model=self.model, dataloaders=self.test_loader)

    def get_model(self) -> pl.LightningModule:
        return self.model

    def get_name(self) -> str:
        return self._name


def run(args: Namespace):
    pl.seed_everything(42, workers=True)

    agg = Noop()

    # setup
    nodes = [
        Node(f"node-{num}", LitCNN()).setup()
        for num in range(args.nodes)
    ]

    # start training
    for round in range(args.rounds):
        for node in nodes:
            print(f"Training {node.get_name()}")
            node.train(args.epochs)

        agg.run([node.get_model() for node in nodes])

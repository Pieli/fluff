from argparse import Namespace

import typing

from models import LitCNN

import torch
from torch.utils import data
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import lightning as pl


def run(args: Namespace):
    pl.seed_everything(42, workers=True)

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

    train_loader = data.DataLoader(
        train_set, batch_size=32, shuffle=True, num_workers=11)
    val_loader = data.DataLoader(
        valid_set, batch_size=32, shuffle=False, num_workers=11)
    test_loader = data.DataLoader(
        training_dataset, batch_size=32, shuffle=False, num_workers=11)

    model = LitCNN()

    trainer = pl.Trainer(max_epochs=args.epochs, deterministic=True)

    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    trainer.test(model=model, dataloaders=test_loader)

import copy
import json
import torch
from torch import nn
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from argparse import Namespace, ArgumentParser
from datetime import datetime

from torch.utils import data

import os
import sys

sys.path.append("../..")

from fluff import Node
from fluff.utils import timer
from fluff.aggregator import FedAvg
from fluff.datasets.cifar10 import CIFAR10Dataset
from fluff.datasets.partitions import DirichletMap, NullMap, BalancedFraction

from typing import cast, Callable, Dict, Any, Optional
from collections.abc import Mapping


import utils
from datasets import CIFAR100Dataset
from models import LitCNN, ServerLitCNNCifar100, CNN, CNNv2
from resnet import ResNet_cifar


def main(args: Namespace):
    pl.seed_everything(args.seed, workers=True)

    print(args)

    def lam_resnet():
        return ResNet_cifar(
            resnet_size=20,
            group_norm_num_groups=2,
            freeze_bn=False,
            freeze_bn_affine=False,
        ).train(True)

    node = Node(
        f"single-model",
        "Single Model Training",
        LitCNN(
            lam_resnet(),
            num_classes=10,
            lr=args.lr,
        ),
        CIFAR10Dataset(
            batch_size=args.batch,
            partition=NullMap(
                partition_id=0,
                partitions_number=1,
            ),
        ),
        num_workers=args.workers,
        hp=args,
    ).setup()

    node.train(
        epochs=args.epochs,
        dev_runs=None,
        skip_val=False,
        skip_test=False,
    )
    node.test()

    os.makedirs(f"./models/", exist_ok=True)
    torch.save(node.get_model().cnn.state_dict(), f"./models/{node.get_name()}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-e", "--epochs", type=int, default=3)
    parser.add_argument("-b", "--batch", type=int, default=128)
    parser.add_argument("-w", "--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--dev-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    main(parser.parse_args())

import torch
from torch import nn
import torchmetrics
import lightning as pl
from argparse import Namespace
from datetime import datetime

from typing import Sequence

import utils
from gradcam import GradCAM
from server_node import (
    lam_cnn,  # noqa: F401
    lam_resnet,
    ServerNode,
)

from models import (
    ServerLitCifar100LogitsOnly,
    ServerLitCNNCifar100,
    ServerLitCifar100InterOnly,
)

from base_trainer import load_models

import sys

sys.path.append("../..")

from fluff import Node
from fluff.utils import timer
from fluff.datasets import CIFAR100Dataset, CIFAR10Dataset
from fluff.datasets.partitions import BalancedFraction
from fluff.aggregator import FedAvg
from fluff.datasets.partitions import DirichletMap


def generate_model_run_name() -> str:
    return f"tiny-Fedours_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"


@timer
def run(args: Namespace):
    pl.seed_everything(args.seed, workers=True)
    exp_name = generate_model_run_name()

    # Training

    print(args)
    ens, stats = load_models(args.base, lam_resnet)

    s_model = lam_resnet()

    for en in ens:
        en.freeze_bn = True  # type: ignore
        en.freeze_bn_affine = True  # type: ignore
        en.train(False)

    # Average

    aggregator = FedAvg()
    s_model.load_state_dict(aggregator.run(ens))

    # Distillation

    server = ServerNode(
        "server",
        exp_name,
        ServerLitCifar100LogitsOnly(
            s_model,
            ensemble=ens,
            distillation=args.distill,
            lr=args.lr,
        ),
        CIFAR100Dataset(
            batch_size=args.batch,
            partition=BalancedFraction(percent=0.8),
            seed=args.seed,
        ),
        num_workers=args.workers,
        seed=args.seed,
        hp=args,
    ).setup()

    print("🧫 Starting distillation")
    server.get_model().set_count_statistics(stats)

    server.train(
        args.rounds,
        dev_runs=args.dev_batches,
        skip_val=False,
        skip_test=False,
    )

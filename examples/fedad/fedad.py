# pylint: disable=import-error

import lightning as pl
from argparse import Namespace
from datetime import datetime


import sys

sys.path.append("../..")

from fluff.utils import timer, plot_tuning
from fluff.datasets import CIFAR100Dataset
from fluff.datasets.partitions import BalancedFraction

from models import ServerLitCNNCifar100

from server_node import (
    lam_cnn,  # noqa: F401
    lam_resnet,
    ServerNode,
)

from base_trainer import load_models


def generate_model_run_name() -> str:
    return f"Fedad_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"


@timer
def run(args: Namespace):
    pl.seed_everything(args.seed, workers=True)
    exp_name = generate_model_run_name()

    # Training
    print("[*]", exp_name)
    print("[+]", args)

    # ens, stats = load_models("./models/five-resnet-alpha-0_5", lam_resnet)
    ens, stats = load_models(args.base, lam_resnet)

    s_model = lam_resnet()

    for en in ens:
        en.freeze_bn = True  # type: ignore
        en.freeze_bn_affine = True  # type: ignore
        en.train(False)

    # Distillation
    server = ServerNode(
        "server",
        exp_name,
        ServerLitCNNCifar100(
            s_model,
            ensemble=ens,
            distillation=args.distill,
        ),
        CIFAR100Dataset(batch_size=args.batch, partition=BalancedFraction(percent=0.8)),
        num_workers=args.workers,
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

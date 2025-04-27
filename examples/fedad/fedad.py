# pylint: disable=import-error

import lightning as pl
from argparse import Namespace
from datetime import datetime
from lightning.pytorch.callbacks import Timer, LambdaCallback


import sys

sys.path.append("../..")

from fluff.utils import timer, UtilizationMonitoring, EarlyStoppingAtTarget
from fluff.datasets import (
    CIFAR100Dataset,
    SVHNDataset,
    CIFAR10Dataset,
    MNISTDataset,
    FMNISTDataset,
    DeepFashionDataset,
)
from fluff.datasets.partitions import BalancedFraction

from models import ServerLitCNNCifar100

from server_node import (
    ServerNode,
    lam_resnet,
    lam_mnist,
)

from base_trainer import load_models


def generate_model_run_name() -> str:
    return f"Fedad_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"


def fedad_fact(set_name: str):
    match set_name:
        case "cifar10":
            return CIFAR100Dataset, CIFAR10Dataset, lam_resnet
        case "mnist":
            return SVHNDataset, MNISTDataset, lam_mnist
        case "fmnist":
            return DeepFashionDataset, FMNISTDataset, lam_mnist


@timer
def run(args: Namespace):
    pl.seed_everything(args.seed, workers=True)
    exp_name = generate_model_run_name()

    # Training
    print("[*]", exp_name)
    print("[+]", args)

    data_cls, base_cls, lam_model = fedad_fact(args.data)

    #  ens, stats = load_models("./models/five-resnet-alpha-0_5", lam_resnet)
    ens, stats = load_models(args.base, lam_model)

    s_model = lam_model()

    timer_call = (
        Timer(duration=dict(minutes=1), interval="step")
        if args.conv
        else LambdaCallback()
    )
    device_stats_callback = UtilizationMonitoring()
    early_stop = (
        EarlyStoppingAtTarget(target_metric="val_f1_epoch", target_value=args.conv)
        if args.conv
        else LambdaCallback()
    )

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
        dataset=data_cls(
            batch_size=args.batch,
            partition=BalancedFraction(percent=0.8),
            seed=args.seed,
        ),
        base_dataset=base_cls(
            BalancedFraction(percent=0.9),
            batch_size=args.batch,
            seed=args.seed,
        ),
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
        callbacks=[timer_call, device_stats_callback, early_stop],
        enable_progress_bar=True,
    )

    if isinstance(early_stop, EarlyStoppingAtTarget):
        if early_stop.stopped:
            print(f"[+] Target reached of f{args.conv}, stopped early")
            return

    if args.conv and timer_call.time_remaining() <= 0.0:
        print("[+] time limit reached...")
        return

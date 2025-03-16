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

from models import LMDServerLitCifar100LogitsOnly

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
    return f"lmd_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"


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

    def adjust_bias(model, stats):
        print(stats)
        print(model.classifier.bias)
        with torch.no_grad():
            bias_adjust = 3.5
            for ind, stat in enumerate(stats):
                if stat == 0:
                    model.classifier.bias[ind] += bias_adjust

    import copy

    copy_s_model = copy.deepcopy(s_model)

    for node in range(args.nodes):

        dataset = CIFAR10Dataset(
            batch_size=args.batch,
            partition=DirichletMap(
                partition_id=node,
                partitions_number=args.nodes,
                alpha=args.alpha,
            ),
            seed=args.seed,
        )

        count = dataset.count_train(num_classes=10)

        print(dataset.count_train())
        cs_model = copy.deepcopy(copy_s_model)
        # adjust_bias(cs_model, dataset.count_train())
        print(cs_model.classifier.bias)

        print(f"[+>] Evaluating node number: {node}")
        server = Node(
            f"server-{node}",
            exp_name,
            LMDServerLitCifar100LogitsOnly(
                cs_model,
                ensemble=ens,
                distillation=args.distill,
                counts=count,
                lr=args.lr,
            ),
            dataset,
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

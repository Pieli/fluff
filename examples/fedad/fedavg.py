import copy
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from argparse import Namespace
from datetime import datetime

import os
import sys

sys.path.append("../..")

from fluff import Node
from fluff.utils import timer
from fluff.aggregator import FedAvg
from fluff.datasets import CIFAR10Dataset
from fluff.datasets.partitions import DirichletMap
from fluff.models import CNN


from typing import Dict, Any
from collections.abc import Mapping


from models import LitCNN
from resnet import ResNet_cifar


def lam_cnn():
    return CNN()
    # return CNNv2()


def lam_resnet():
    return ResNet_cifar(
        resnet_size=20,
        group_norm_num_groups=2,
        freeze_bn=False,
        freeze_bn_affine=False,
    ).train(True)


def generate_model_run_name() -> str:
    return f"FedAvg_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"


class MyStrat(pl.pytorch.strategies.SingleDeviceStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model_state_dict(
        self, checkpoint: Mapping[str, Any], strict: bool = True
    ) -> None:
        pass


@timer
def run(args: Namespace):

    pl.seed_everything(args.seed, workers=True)
    exp_name = generate_model_run_name()

    agg = FedAvg()
    model_type = lam_resnet

    print(args)

    nodes = [
        Node(
            f"node-{num}",
            exp_name,
            LitCNN(
                model_type(),
                num_classes=10,
                lr=1e-3,
            ),
            CIFAR10Dataset(
                batch_size=args.batch,
                partition=DirichletMap(
                    partition_id=num,
                    partitions_number=args.nodes,
                    alpha=args.alpha,
                ),
            ),
            num_workers=args.workers,
            hp=args,
        ).setup()
        for num in range(args.nodes)
    ]

    # For the evaluation of the global model
    # even though a partition is loaded it is not needed since the test set
    # stays unchanged, could change DirichletMap to NullMap
    server = Node(
        "server",
        exp_name,
        LitCNN(
            model_type(),
            num_classes=10,
            lr=1e-3,
        ),
        CIFAR10Dataset(
            batch_size=args.batch,
            partition=DirichletMap(
                partition_id=0,
                partitions_number=args.nodes,
            ),
        ),
        num_workers=args.workers,
    ).setup()

    callback = [ModelCheckpoint(save_last=True) for _ in range(args.nodes)]
    chk_point = f"./fluff_logs/{exp_name}"
    paths = [
        os.path.join(
            chk_point,
            node.get_name(),
            "LitCNN",
            "checkpoints",
            "last.ckpt",
        )
        for node in nodes
    ]

    new_state: Dict[str, Any] = {}
    for round in range(args.rounds):
        print(f"[+] Started round number {round + 1}")
        for ind, node in enumerate(nodes):
            print(f"[+] Started training node {node.get_name()} - Round {round + 1}")

            if round > 0:
                node._model.cnn.load_state_dict(new_state)

            node.train(
                epochs=args.epochs * (round + 1),
                dev_runs=args.dev_batches,
                skip_val=False,
                skip_test=False,
                callbacks=[callback[ind]],
                ckpt_path=(paths[ind] if round > 0 else None),
                strat=MyStrat(device="cuda:0"),
                enable_progress_bar=False,
            )

        new_state = agg.run([node.get_model().cnn for node in nodes])
        server._model.cnn.load_state_dict(copy.deepcopy(new_state))
        server.test()

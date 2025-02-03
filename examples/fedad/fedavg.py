import json
import torch
from torch import nn
import lightning as pl
from argparse import Namespace
from datetime import datetime

from torch.utils import data

import os
import sys
sys.path.append('../..')

from fluff import Node
from fluff.utils import timer
from fluff.aggregator import FedAvg
from fluff.datasets.cifar10 import CIFAR10Dataset
from fluff.datasets.partitions import DirichletMap, NullMap, BalancedFraction

from typing import cast, Callable, Dict, Any, Optional
from collections.abc import Mapping


import utils
from datasets import CIFAR100Dataset
from models import LitCNN, ServerLitCNNCifar100, CNN
from resnet import ResNet_cifar


def lam_cnn():
    return CNN(num_classes=10)


def lam_resnet():
    return ResNet_cifar(resnet_size=20, group_norm_num_groups=2,
                        freeze_bn=False, freeze_bn_affine=False,).train(True)


def generate_model_run_name() -> str:
    return f"Fedad_{datetime.now().strftime("%d-%m-%Y_%H:%M:%S")}"


def compare_models(model_1: Dict[str, Any], model_2: Dict[str, Any]):
    models_differ = False

    # Ensure both models have the same keys
    if set(model_1.keys()) != set(model_2.keys()):
        print("Mismatch in keys between the two models!")
        print(set(model_1.keys()), set(model_2.keys()))
        return

    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        key_1, value_1 = key_item_1
        key_2, value_2 = key_item_2

        # Check if the key names are the same (they should be by default)
        if key_1 == key_2:
            # Compare the tensors for near-equality
            # You can adjust atol (absolute tolerance)
            if not torch.allclose(value_1.cpu(), value_2.cpu(), atol=1e-5):
                models_differ = True
                print(f"Mismatch found at {key_1}")
                print(
                    f"Max difference: {(value_1.cpu() - value_2.cpu()).abs().max().item()}")
        else:
            print(f"Keys mismatch: {key_1} != {key_2}")
            models_differ = True

    if not models_differ:
        print("Models match perfectly! :)")


class MyStrat(pl.pytorch.strategies.SingleDeviceStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict: bool = True) -> None:
        print("HELLLLOOOOOOOOO")
        pass


@timer
def run(args: Namespace):

    pl.seed_everything(args.seed, workers=True)
    exp_name = generate_model_run_name()

    agg = FedAvg()
    model_type = lam_cnn
    """
        partition=DirichletMap(
            partition_id=num,
            partitions_number=args.nodes
        )),
        """

    nodes = [
        Node(f"node-{num}",
             exp_name,
             LitCNN(model_type(), num_classes=10),
             CIFAR10Dataset(
                 batch_size=args.batch,
                 partition=BalancedFraction(0.1)),
             num_workers=args.workers
             ).setup() for num in range(args.nodes)
    ]

    from lightning.pytorch.callbacks import ModelCheckpoint

    callback = [ModelCheckpoint(save_last=True) for _ in range(args.nodes)]
    chk_point = f"./fluff_logs/{exp_name}"
    paths = [os.path.join(chk_point,
                          node.get_name(),
                          "LitCNN",
                          "checkpoints",
                          "last.ckpt") for node in nodes]

    agg_models = []
    import copy

    new_state: Dict[str, Any] = {}
    for round in range(args.rounds):
        print(f"[+] Started round number {round + 1}")
        for ind, node in enumerate(nodes):
            print(
                f"[+] Started training node {node.get_name()} - Round {round + 1}")

            if round > 0:
                node._model.cnn.load_state_dict(new_state)
                agg_models.append(copy.deepcopy(node._model.cnn.state_dict()))

                print("hello")
                compare_models(agg_models[ind], new_state)

            node.train(epochs=args.epochs * (round + 1),
                       dev_runs=args.dev_batches,
                       skip_val=True,
                       callbacks=[callback[ind]],
                       ckpt_path=(paths[ind] if round > 0 else None),
                       strat=MyStrat(device="cuda:0"))

        new_state = agg.run([node.get_model().cnn for node in nodes])

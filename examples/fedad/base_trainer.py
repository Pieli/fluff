import json
import torch
from torch import nn
from argparse import Namespace

import lightning as pl

import os
import sys

sys.path.append("../..")

from fluff import Node
from fluff.datasets import CIFAR10Dataset, MNISTDataset, FMNISTDataset
from fluff.datasets.partitions import DirichletMap
from fluff.models import CNNv2

from typing import cast, Callable

from models import LitCNN
from resnet import ResNet_cifar, ResNet_mnist


def lam_resnet():
    return (
        ResNet_cifar(
            resnet_size=20,
            group_norm_num_groups=2,
            freeze_bn=False,
            freeze_bn_affine=False,
        ).train(True),
    )


def lam_mnist():
    return ResNet_mnist(
        resnet_size=20,
        group_norm_num_groups=2,
        freeze_bn=True,
        freeze_bn_affine=True,
    ).train(True)


def fact(set_name):
    match set_name:
        case "cifar10":
            return CIFAR10Dataset, lam_resnet
        case "mnist":
            return MNISTDataset, lam_mnist
        case "fmnist":
            return FMNISTDataset, lam_mnist


def training_phase(
    args: Namespace, name: str, save=False
) -> tuple[list[nn.Module], list[torch.Tensor]]:

    pl.seed_everything(args.seed, workers=True)

    # split out ./models
    name = os.path.split(name)[1]

    set_cls, lam_model = fact(args.data)

    node_stats = {}
    nodes: list[Node] = []
    for num in range(args.nodes):
        node_cifar10 = Node(
            f"node-{num}",
            name,
            LitCNN(
                lam_model(),
                num_classes=10,
            ),
            set_cls(
                batch_size=16,
                partition=DirichletMap(
                    partition_id=num,
                    partitions_number=args.nodes,
                    alpha=args.alpha,
                ),
            ),
            seed=args.seed,
            num_workers=args.workers,
            hp=args,
        ).setup()

        # start training
        print(f"Training  - {node_cifar10.get_name()}")

        # 200 epochs
        node_cifar10.train(
            args.epochs,
            args.dev_batches,
            swap_val_to_test=args.use_test_loader,
            skip_val=False,
            skip_test=False,
        )

        nodes.append(node_cifar10)
        node_stats[node_cifar10.get_name()] = (
            node_cifar10.get_model().get_statistics().tolist()
        )

    if save:
        os.makedirs(f"./models/{name}", exist_ok=True)
        for node in nodes:
            print(f"Saving model ({node.get_name()}) ...")
            torch.save(
                node.get_model().cnn.state_dict(), f"./models/{name}/{node.get_name()}"
            )

        with open(f"./models/{name}/statistics", "w") as f:
            json.dump(node_stats, f)

    stats = [node.get_model().get_statistics().unsqueeze(dim=1) for node in nodes]
    return [node.get_model().cnn for node in nodes], stats


def load_stats(path: str, model_names: list[str]) -> list[torch.Tensor]:
    assert os.path.isfile(path)
    with open(path, "r") as f:
        result = json.load(f)

    return list(torch.tensor(result[mod]).unsqueeze(dim=1) for mod in model_names)


def load_models(
    dir: str,
    model: Callable,
) -> tuple[list[nn.Module], list[torch.Tensor]]:

    if not os.path.exists(dir):
        print(f"[-] Path {dir} does not exist")
        exit(1)

    names = sorted(os.listdir(dir))

    if "statistics" in names:
        names.remove("statistics")

    nodes = []
    for name in names:
        path = f"{dir}/{name}"
        if not os.path.isfile(path):
            print(f"File {path} was not found (skipped)")
            continue
        node = model()
        node.load_state_dict(torch.load(path, weights_only=True))
        nodes.append(cast(nn.Module, node))

    print(f"[+] Successfully loaded {dir}")
    return nodes, load_stats(f"{dir}/statistics", names)


def train(args: Namespace):
    training_phase(args, name=args.base, save=True)

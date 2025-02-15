import json
import torch
from torch import nn
import lightning as pl
from argparse import Namespace
from datetime import datetime

from torch.utils import data

import os
import sys

sys.path.append("../..")

from fluff import Node
from fluff.utils import timer
from fluff.aggregator import FedAvg
from fluff.datasets import CIFAR10Dataset, CIFAR100Dataset
from fluff.datasets.partitions import DirichletMap, NullMap, BalancedFraction

from typing import cast, Callable


import utils
from models import LitCNN, ServerLitCNNCifar100, CNN
from resnet import ResNet_cifar


class ServerNode(Node):
    def __init__(
        self,
        name: str,
        experiement_name: str,
        model: pl.LightningModule,
        dataset,
        num_workers: int = 2,
        seed=None,
        hp=None,
    ) -> None:
        super().__init__(name, experiement_name, model, dataset, num_workers, seed, hp)

    def setup(self):
        cif_10 = CIFAR10Dataset(
            BalancedFraction(percent=0.9), batch_size=self._dataset.get_batch_size()
        )

        training_dataset = data.Subset(
            self._dataset.train_set,
            self._dataset.train_indices_map,
        )

        train_set = training_dataset
        _, valid_set = data.random_split(
            cif_10.train_set, (0.8, 0.2), generator=torch.Generator().manual_seed(42)
        )
        test_set = cif_10.test_set

        generator = torch.Generator().manual_seed(self._seed) if self._seed else None
        self.train_loader = data.DataLoader(
            train_set,
            batch_size=self._dataset.get_batch_size(),
            shuffle=True,
            num_workers=self._num_workers,
            generator=generator,
        )

        self.val_loader = data.DataLoader(
            valid_set,
            batch_size=self._dataset.get_batch_size(),
            shuffle=False,
            num_workers=self._num_workers,
        )

        self.test_loader = data.DataLoader(
            test_set,
            batch_size=self._dataset.get_batch_size(),
            shuffle=False,
            num_workers=self._num_workers,
        )
        return self


def persist_configuration(args: Namespace, file_name: str):
    with open(file_name, "w") as f:
        f.writelines(f"{name}\t{value}\n" for (name, value) in args._get_kwargs())


def persist_model(model: nn.Module, dir: str):
    pass


def generate_model_run_name() -> str:
    return f"Fedad_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"


def training_phase(
    args: Namespace, name: str, save=False
) -> tuple[list[nn.Module], list[torch.Tensor]]:
    node_stats = {}
    nodes: list[Node] = []
    for num in range(args.nodes):
        node_cifar10 = Node(
            f"node-{num}",
            name,
            LitCNN(
                ResNet_cifar(
                    resnet_size=20,
                    group_norm_num_groups=2,
                    freeze_bn=False,
                    freeze_bn_affine=False,
                ).train(True),
                num_classes=10,
            ),
            CIFAR10Dataset(
                batch_size=16,
                partition=DirichletMap(
                    partition_id=num,
                    partitions_number=args.nodes,
                    alpha=args.alpha,
                ),
            ),
            num_workers=args.workers,
            hp=args,
        ).setup()

        # start training cifar10
        print(f"Training CIFAR10 - {node_cifar10.get_name()}")

        # 200 epochs for CIFAR10
        node_cifar10.train(
            args.epochs,
            args.dev_batches,
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


def load_stats(path: str) -> list[torch.Tensor]:
    assert os.path.isfile(path)
    with open(path, "r") as f:
        result = json.load(f)

    return list(torch.tensor(elem).unsqueeze(dim=1) for elem in result.values())


def load_models(
    dir: str,
    model: Callable,
) -> tuple[list[nn.Module], list[torch.Tensor]]:
    names = os.listdir(dir)

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
    return nodes, load_stats(f"{dir}/statistics")


def lam_cnn():
    return CNN(num_classes=10)


def lam_resnet():
    return ResNet_cifar(
        resnet_size=20,
        group_norm_num_groups=2,
        freeze_bn=False,
        freeze_bn_affine=False,
    ).train(True)


@timer
def run(args: Namespace):
    pl.seed_everything(args.seed, workers=True)
    exp_name = generate_model_run_name()

    # Training
    print(args)
    # ens, stats = training_phase(args, name="five-resnet-alpha-0_01", save=True)
    ens, stats = load_models("./models/five-resnet-alpha-0_5", lam_resnet)

    s_model = lam_resnet()

    for en in ens:
        en.freeze_bn = True  # type: ignore
        en.freeze_bn_affine = True  # type: ignore
        en.train(False)

    # aggregator = FedAvg()
    # s_model.load_state_dict(aggregator.run(ens))

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

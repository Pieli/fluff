from torch import nn
import lightning as pl
from argparse import Namespace
from datetime import datetime

import sys
sys.path.append('../..')

from fluff import Node
from fluff.utils import timer
from fluff.datasets.cifar10 import CIFAR10Dataset
from fluff.datasets.partitions import DirichletMap, NullMap, BalancedFraction


import utils
from datasets import CIFAR100Dataset
from models import LitCNN, LitCNN_Cifar100, ServerLitCNNCifar100, CNN


def persist_configuration(args: Namespace, file_name: str):
    with open(file_name, "w") as f:
        f.writelines(
            f"{name}\t{value}\n"
            for (name, value) in args._get_kwargs()
        )


def persist_model(model: nn.Module, dir: str):
    pass


def generate_model_run_name() -> str:
    return f"Fedad_{datetime.now().strftime("%d-%m-%Y_%H:%M:%S")}"


@timer
def run(args: Namespace):
    pl.seed_everything(args.seed, workers=True)
    exp_name = generate_model_run_name()
    # persist_configuration(args, "config.")

    # Training
    print("🚀 Starting training")
    nodes = []
    for num in range(args.nodes):

        node_cifar10 = Node(f"node-{num}",
                            exp_name,
                            LitCNN(),
                            CIFAR10Dataset(
                                batch_size=16,
                                partition=DirichletMap(
                                    partition_id=num,
                                    partitions_number=args.nodes
                                )),
                            num_workers=args.workers
                            ).setup()

        # start training cifar10
        print(f"Training CIFAR10 - {node_cifar10.get_name()}")

        # 200 epochs for CIFAR10
        node_cifar10.train(args.epochs, args.dev_batches)
        node_cifar10.test(args.epochs, args.dev_batches)

        # change model
        model = node_cifar10.get_model()

        for param in model.cnn.parameters():
            param.requires_grad = False

        model.cnn.fc2 = nn.Linear(64, 100)
        model.cnn.fc2.requires_grad = True
        model.cnn.fc2.bias.requires_grad = True

        # set node to new model + dataset
        node_cifar100 = Node(f"node-{num}",
                             exp_name,
                             LitCNN_Cifar100(model.cnn),
                             CIFAR100Dataset(
                                 batch_size=16,
                                 partition=NullMap(
                                     partition_id=num,
                                     partitions_number=args.nodes
                                 ))).setup()

        nodes.append(node_cifar100)

        print(f"Training CIFAR100 - {node_cifar100.get_name()}")
        node_cifar100.train(1, args.dev_batches)
        node_cifar100.test(args.epochs, args.dev_batches)

    nodes = [node.get_model().cnn for node in nodes]

    server = Node("server",
                  exp_name,
                  ServerLitCNNCifar100(
                      CNN(num_classes=100),
                      distillation_phase=False,
                      ensemble=nodes,
                  ),
                  CIFAR100Dataset(
                      batch_size=512,
                      partition=BalancedFraction(percent=0.1)),
                  ).setup()

    print("\N{Flexed Biceps} Pre-Training server")
    server.train(1, args.dev_batches)  # TODO change 1 -> 10

    print("🧫 Starting distillation")
    server.get_model().set_distillation_phase(True)
    server.train(1, None)
    server.test(1, None)

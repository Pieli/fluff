from torch import nn
from torch.utils import data
import lightning as pl
from argparse import Namespace

import sys
sys.path.append('../..')

from fluff import Node
from fluff.utils import timer
from fluff.datasets.cifar10 import CIFAR10Dataset
from fluff.datasets.partitions.dirichelet_map import DirichletMap
from fluff.datasets.partitions.balanced_iid_map import BalancedIIDMap


import utils
from datasets import CIFAR100Dataset
from models import LitCNN, LitCNN_Cifar100, CNN


@timer
def run(args: Namespace):
    pl.seed_everything(42, workers=True)

    node = Node("node-0",
                LitCNN_Cifar100(CNN(num_classes=100), distillation=True),
                CIFAR10Dataset(
                    batch_size=200,
                    partition=DirichletMap(
                        partition_id=0,
                        partitions_number=args.nodes
                    ))).setup()
    print(repr(node))
    node.train(2, False)
    return node
    exit(0)

    # Training
    nodes = []
    for num in range(args.nodes):

        # TODO
        # - init server node
        # - cosine annealing
        # - how many epochs for test
        # - configure cifar100

        node_cifar10 = Node(f"node-{num}",
                            LitCNN(),
                            CIFAR10Dataset(
                                batch_size=16,
                                partition=DirichletMap(
                                    partition_id=num,
                                    partitions_number=args.nodes
                                ))).setup()

        nodes.append(node_cifar10)

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
                             LitCNN_Cifar100(model.cnn),
                             CIFAR100Dataset(
                                 batch_size=16,
                                 partition=BalancedIIDMap(
                                     partition_id=num,
                                     partitions_number=args.nodes
                                 ))).setup()

        nodes[num] = node_cifar100
        # del node_cifar10

        print(f"Training CIFAR100 - {node_cifar100.get_name()}")
        # 10 epochs for CIFAR100
        node_cifar100.train(args.epochs, args.dev_batches)
        node_cifar100.test(args.epochs, args.dev_batches)

    # Distillation rounds
    for round in range(args.rounds):
        for node in nodes:
            print(f"Training {node.get_name()}")
            node.train(args.epochs, args.dev_batches)

from torch import nn
import lightning as pl
from argparse import Namespace

import sys
sys.path.append('../..')

from fluff import Node
from fluff.utils import timer
from fluff.datasets.cifar10 import CIFAR10Dataset
from fluff.datasets.partitions import DirichletMap, NullMap


import utils
from datasets import CIFAR100Dataset
from models import LitCNN, LitCNN_Cifar100, ServerLitCNNCifar100


@timer
def run(args: Namespace):
    pl.seed_everything(42, workers=True)

    # Training
    print("🚀 Starting training")
    nodes = []
    for num in range(args.nodes):

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
                             LitCNN_Cifar100(model.cnn, distillation=False),
                             CIFAR100Dataset(
                                 batch_size=16,
                                 partition=NullMap(
                                     partition_id=num,
                                     partitions_number=args.nodes
                                 ))).setup()

        nodes[num] = node_cifar100
        # del node_cifar10

        print(f"Training CIFAR100 - {node_cifar100.get_name()}")
        # 10 epochs for CIFAR100
        node_cifar100.train(args.epochs, args.dev_batches)
        node_cifar100.test(args.epochs, args.dev_batches)

    nodes = [Node(node.get_name(),
                  node.get_model().set_distillation(True),
                  CIFAR100Dataset(batch_size=16,
                                  partition=NullMap(
                                      partition_id=0,
                                      partitions_number=1
                                  )),
                  seed=420).setup()
             for node in nodes]

    server = Node("server",
                  ServerLitCNNCifar100(model.cnn, distillation_phase=False),
                  CIFAR100Dataset(
                      batch_size=16,
                      partition=NullMap(
                          partition_id=0,
                          partitions_number=1
                      )),
                  seed=420).setup()
    # TODO
    # distillation method
    # pretrain server

    print("\N{Flexed Biceps} Pre-Training server")
    server.train(args.epochs, args.dev_batches)

    print("🧫 Starting distillation")
    server.get_model().set_distillation_phase(True)
    for round in range(args.rounds):
        round_logits = []
        round_counts = []
        for node in nodes:
            print(f"Training {node.get_name()}")
            node.train(1, args.dev_batches)
            round_logits.append(node.get_model().get_average_logits())
            round_counts.append(node.get_model().get_class_counts())

        batch_logits = zip(*round_logits)
        batch_counts = zip(*round_counts)

        ens_logits = [utils.logits_ensemble_eq_3(log, count, 100, args.nodes)
                      for log, count in zip(batch_logits, batch_counts)]

        server.get_model().set_ensemble_logits(ens_logits)
        server.train(1, args.dev_batches)
        server.test(1, args.dev_batches)
        # server.distill(nodes)
        print(round_logits)

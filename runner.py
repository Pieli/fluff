import lightning as pl
from argparse import Namespace

from .. import fluff

from fluff import aggregator

from fluff.models import LitCNN
from fluff.utils import timer
from fluff.datasets.cifar10 import CIFAR10Dataset
from fluff.datasets.partitions.dirichelet_map import DirichletMap


@timer
def run(args: Namespace):
    pl.seed_everything(42, workers=True)

    agg = aggregator.FedAvg()

    # setup
    nodes = [
        fluff.Node(f"node-{num}", LitCNN(), CIFAR10Dataset(DirichletMap(partition_id=num, partitions_number=args.nodes))).setup()
        for num in range(args.nodes)
    ]

    # start training
    for round in range(args.rounds):
        for node in nodes:
            print(f"Training {node.get_name()}")
            node.train(args.epochs, args.dev_batches)

        agg.run([node.get_model() for node in nodes])

    nodes[0].test(args.epochs, args.dev_batches)

import torch
from torch.utils import data
import lightning as pl
from argparse import Namespace

from utils import timer
from models import LitCNN
import aggregator
from datasets.dataset import NebulaDataset
from datasets.cifar10Improved import CIFAR10Dataset
from datasets.partitions.dirichelet_map import DirichletMap


class Node:
    def __init__(self, name: str,
                 model: pl.LightningModule,
                 dataset: NebulaDataset,
                 num_workers: int = 2) -> None:

        self._name = name
        self._model = model
        self._dataset = dataset
        self._num_workers = num_workers

    def setup(self):
        training_dataset = data.Subset(self._dataset.train_set, self._dataset.train_indices_map)
        test_dataset = self._dataset.test_set

        train_set_size = int(len(training_dataset) * 0.8)
        valid_set_size = len(training_dataset) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)

        train_set, valid_set = data.random_split(
            training_dataset, [train_set_size, valid_set_size], generator=seed)

        self.train_loader = data.DataLoader(
            train_set, batch_size=32, shuffle=True, num_workers=self._num_workers)

        self.val_loader = data.DataLoader(
            valid_set, batch_size=32, shuffle=False, num_workers=self._num_workers)

        self.test_loader = data.DataLoader(
            test_dataset, batch_size=32, shuffle=False, num_workers=self._num_workers)

        return self

    def train(self, epochs: int, dev_runs=False) -> None:
        trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=dev_runs, deterministic=True)

        trainer.fit(model=self._model,
                    train_dataloaders=self.train_loader,
                    val_dataloaders=self.val_loader)

    def test(self, epochs: int, dev_runs=False) -> None:
        trainer = pl.Trainer(max_epochs=epochs, fast_dev_run=dev_runs, deterministic=True)
        trainer.test(model=self._model, dataloaders=self.test_loader)

    def get_model(self) -> pl.LightningModule:
        return self._model

    def get_name(self) -> str:
        return self._name


@timer
def run(args: Namespace):
    pl.seed_everything(42, workers=True)

    agg = aggregator.FedAvg()

    # setup
    nodes = [
        Node(f"node-{num}", LitCNN(), CIFAR10Dataset(DirichletMap(partition_id=num, partitions_number=args.nodes))).setup()
        for num in range(args.nodes)
    ]

    # start training
    for round in range(args.rounds):
        for node in nodes:
            print(f"Training {node.get_name()}")
            node.train(args.epochs, args.dev_batches)

        agg.run([node.get_model() for node in nodes])

    nodes[0].test(args.epochs, args.dev_batches)

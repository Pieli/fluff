from models import LitCNN

from abc import ABC, abstractmethod
from torch.utils import data
from argparse import ArgumentParser


parser = ArgumentParser()

parser.add_argument("--nodes", type=int, default=2)
parser.add_argument("--local_batch", type=int, default=128)
parser.add_argument("--rounds", type=int, default=128)

# args = parser.parse_args()


class Config:
    pass


class Dataset(data.Dataset, ABC):
    def __init__(self, dataset):
        pass

    @abstractmethod
    def load(self):
        pass

    def plot(self):
        pass


# balanced
# unbalanced
# dataset = BalancedIID(Dataset()).load()


if __name__ == "__main__":

    trainer = Trainer(devices=args.devices)
    model = MyModel(layer_1_dim=args.layer_1_dim)

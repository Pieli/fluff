import sys
sys.path.append("..")

from fluff.datasets.partitions import BalancedFraction
from fluff.datasets.cifar10 import CIFAR10Dataset


def test_fraction():

    dataset = CIFAR10Dataset(batch_size=100,
                             partition=BalancedFraction(percent=0.7))

    assert len(dataset.train_indices_map) == 0.7 * 50_000

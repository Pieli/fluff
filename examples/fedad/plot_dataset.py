import sys

sys.path.append("../..")


from fluff.datasets import CIFAR10Dataset
from fluff.datasets.partitions import DirichletMap

dataset = CIFAR10Dataset(
    batch_size=64,
    partition=DirichletMap(
        partition_id=0,
        partitions_number=5,
        alpha=0.5,
    ),
)


dataset.plot()

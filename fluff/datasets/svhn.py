import os
from .partitions.partitions import Partition
from .dataset import NebulaDataset

from torchvision import transforms
from torchvision.datasets import SVHN


class SVHNDataset(NebulaDataset):
    def __init__(
        self,
        partition: Partition,
        batch_size=32,
        num_classes=10,
        num_workers=4,
        seed=42,
    ):
        super().__init__(
            partition=partition,
            num_classes=num_classes,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
        )

    def initialize_dataset(self):
        # Load CIFAR10 train dataset
        if self.train_set is None:
            self.train_set = self.load(train=True)

        if self.test_set is None:
            self.test_set = self.load(train=False)

        # to add the missing but required targets attribute
        self.train_set.targets = self.train_set.labels
        self.test_set.targets = self.test_set.labels

        # All nodes have the same test set (indices are the same for all nodes)
        self.test_indices_map = list(range(len(self.test_set)))

        # Depending on the partition class non-iid or iid partions are created.
        self.partition.set_seed(self.seed)
        self.partition.set_num_classes(self.num_classes)
        self.partition_map = self.partition.generate(self.train_set)
        self.train_indices_map = self.partition_map[self.partition.get_id()]
        self.local_test_indices_map = self.partition.generate(self.test_set)[
            self.partition.get_id()
        ]

        print(f"Len of train indices map (global): {len(self.train_set)}")
        print(f"Len of train indices map (local)): {len(self.train_indices_map)}")
        print(f"Len of test indices map (global): {len(self.test_indices_map)}")
        print(f"Len of test indices map (local): {len(self.local_test_indices_map)}")

    def load(self, train=True):
        apply_transforms = transforms.Compose(
            [
                transforms.Resize((28, 28)),  # Resize to MNIST size
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize((0.4514,), (0.1987,)),
            ]
        )
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        return SVHN(
            data_dir,
            split="train" if train else "test",
            download=True,
            transform=apply_transforms,
        )

    def plot(self):
        self.partition.plot_all_data_distribution(self.train_set, self.partition_map)
        self.partition.plot_data_distribution(self.train_set, self.partition_map)

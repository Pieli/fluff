import os
from torchvision import transforms
from torchvision.datasets import CIFAR100

from fluff.datasets.dataset import NebulaDataset
from fluff.datasets.partitions.partitions import Partition


class CIFAR100Dataset(NebulaDataset):
    def __init__(
        self,
        partition: Partition,
        batch_size=32,
        num_classes=1000,
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
        self.train_set = self.load_cifar100_dataset(train=True)
        self.test_set = self.load_cifar100_dataset(train=False)

        # All nodes have the same test set (indices are the same for all nodes)
        self.test_indices_map = list(range(len(self.test_set)))

        # Depending on the partition class non-iid or iid partions are created.
        self.partition.set_seed(self.seed)
        self.partition.set_num_classes(self.num_classes)
        self.partition_map = self.partition.generate(self.train_set)
        self.train_indices_map = self.partition_map[self.partition.get_id()]
        self.local_test_indices_map = self.partition.generate(self.test_set)[self.partition.get_id()]

        print(f"Len of train indices map: {len(self.train_indices_map)}")
        print(
            f"Len of test indices map (global): {len(self.test_indices_map)}")
        print(
            f"Len of test indices map (local): {len(self.local_test_indices_map)}")

    def load_cifar100_dataset(self, train=True):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        apply_transforms = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        return CIFAR100(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
            train=train,
            download=True,
            transform=apply_transforms,
        )

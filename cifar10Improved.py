from partitions import Partition
from dataset import NebulaDataset


class CIFAR10Dataset(NebulaDataset):
    def __init__(
        self,
        partition: Partition,
        num_classes=10,
        batch_size=32,
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
            self.train_set = self.load_cifar10_dataset(train=True)

        if self.test_set is None:
            self.test_set = self.load_cifar10_dataset(train=False)

        # All nodes have the same test set (indices are the same for all nodes)
        self.test_indices_map = list(range(len(self.test_set)))

        # Depending on the partition class non-iid or iid partions are created.
        self.train_indices_map = self.partition.generate(self.train_set)
        self.local_test_indices_map = self.partition.generate(self.test_set)

        print(f"Lenof train indices map: {len(self.train_indices_map)}")
        print(
            f"Len of test indices map (global): {len(self.test_indices_map)}")
        print(
            f"Len of test indices map (local): {len(self.local_test_indices_map)}")

    def load_cifar10_dataset(self, train=True, override=False):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2471, 0.2435, 0.2616)
        apply_transforms = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std, inplace=True),
        ])
        data_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        return CIFAR10(
            data_dir,
            train=train,
            download=True,
            transform=apply_transforms,
        )

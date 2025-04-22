import os
import pathlib

from .partitions.partitions import Partition
from .dataset import NebulaDataset

from torchvision import transforms

from PIL import Image
from torch.utils.data import Dataset


"""

Note: This is only a temporary implementation of the DeepFashion Dataset

"""


class DFDataset(Dataset):
    def __init__(self, root_directory, transform=None):
        root_path = pathlib.Path(root_directory)
        if not root_path.exists():
            raise ValueError(f"{root_path} folder does not exist")

        self.image_folder = root_path / "img"
        if not self.image_folder.exists():
            raise ValueError(f"{self.image_folder} does not exist")

        self.train_imgs = root_path / "train.txt"
        if not self.train_imgs.exists():
            raise ValueError(f"{self.train_imgs} does not exist")

        self.train_targets = root_path / "train_cate.txt"
        if not self.train_targets.exists():
            raise ValueError(f"{self.train_targets} does not exist")

        self.transform = transform
        self.targets = []
        self.image_labels = []

        with open(self.train_targets, 'r') as f:
            self.targets = [
                int(line.strip())
                for line in f.readlines()
            ]

        with open(self.train_imgs, 'r') as f:
            for path_line, label in zip(f.readlines(), self.targets):
                path = path_line.strip()
                self.image_labels.append((path, int(label)))

    def __len__(self):
        return len(self.image_labels)

    def __getitem__(self, idx):
        img_path, label = self.image_labels[idx]
        image = Image.open(os.path.join(
            self.image_folder, img_path)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class DeepFashionDataset(NebulaDataset):
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

        # Depending on the partition class non-iid or iid partions are created.
        self.partition.set_seed(self.seed)
        self.partition.set_num_classes(self.num_classes)
        self.partition_map = self.partition.generate(self.train_set)
        self.train_indices_map = self.partition_map[self.partition.get_id()]

        print(f"Len of train indices map (global): {len(self.train_set)}")
        print(
            f"Len of train indices map (local)): {len(self.train_indices_map)}")

    def load(self, train=True):
        apply_transforms = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.7451], std=[0.2218]),
            ]
        )
        data_dir = pathlib.Path(__file__).parent / "data" / "deepfashion"
        return DFDataset(
            data_dir,
            transform=apply_transforms,
        )

    def plot(self):
        self.partition.plot_all_data_distribution(
            self.train_set, self.partition_map)
        self.partition.plot_data_distribution(
            self.train_set, self.partition_map)

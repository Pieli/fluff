import logging
import numpy as np
from abc import ABC, abstractmethod
import seaborn as sns
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict

from torch.utils import data

matplotlib.use("Agg")
plt.switch_backend("Agg")


class Partition(ABC):
    def __init__(self, partition_id: int, partitions_number: int, partition_parameter: float):
        self._id = partition_id
        self._number = partitions_number
        self._parameter = partition_parameter
        self.seed = 42

        self.class_distribution = None

        if partition_id < 0 or partition_id >= partitions_number:
            raise ValueError(
                f"partition_id {partition_id} is out of range for partitions_number {partitions_number}")

    def get_id(self):
        return self._id

    def set_seed(self, seed: int):
        self.seed = seed

    def set_num_classes(self, num: int):
        self.num_classes = num

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def is_iid(self) -> bool:
        pass

    @abstractmethod
    def generate(self, dataset: data.Dataset, **kwargs):
        pass

    def plot_data_distribution(self, dataset: data.Dataset, partitions_map):
        """
        Plot the data distribution of the dataset.

        Plot the data distribution of the dataset according to the partitions map provided.

        Args:
            dataset: The dataset to plot (torch.utils.data.Dataset).
            partitions_map: The map of the dataset partitions.
        """
        # Plot the data distribution of the dataset, one graph per partition
        sns.set()
        sns.set_style("whitegrid", {"axes.grid": False})
        sns.set_context("paper", font_scale=1.5)
        sns.set_palette("Set2")

        for i in range(self._number):
            indices = partitions_map[i]
            class_counts = [0] * self.num_classes
            for idx in indices:
                label = dataset.targets[idx]
                class_counts[label] += 1
            logging.info(
                f"Participant {i + 1} class distribution: {class_counts}")
            plt.figure()
            plt.bar(range(self.num_classes), class_counts)
            plt.xlabel("Class")
            plt.ylabel("Number of samples")
            plt.xticks(range(self.num_classes))
            plt.title(
                f"Participant {i + 1} class distribution ({self.get_name()} - {self._parameter})"
            )
            plt.tight_layout()
            path_to_save = f"./participant_{i}_class_distribution_{'iid' if self.is_iid() else 'non_iid'}{'_' + self.get_name()}.png"
            plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
            plt.close()

        plt.figure()
        max_point_size = 500
        min_point_size = 0

        for i in range(self._number):
            class_counts = [0] * self.num_classes
            indices = partitions_map[i]
            for idx in indices:
                label = dataset.targets[idx]
                class_counts[label] += 1

            # Normalize the point sizes for this partition
            max_samples_partition = max(class_counts)
            sizes = [
                (size / max_samples_partition) *
                (max_point_size - min_point_size) + min_point_size
                for size in class_counts
            ]
            plt.scatter([i] * self.num_classes,
                        range(self.num_classes), s=sizes, alpha=0.5)

        plt.xlabel("Participant")
        plt.ylabel("Class")
        plt.xticks(range(self._number))
        plt.yticks(range(self.num_classes))
        if self.is_iid():
            plt.title(f"Participant {i + 1} class distribution (IID)")
        else:
            plt.title(
                f"Participant {i + 1} class distribution (Non-IID - {self.get_name()}) - {self._parameter}"
            )
        plt.tight_layout()

        # Saves the distribution display with circles of different size
        path_to_save = f"./class_distribution_{'iid' if self.is_iid() else 'non_iid'}{'_' + self.get_name()}.png"
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
        plt.close()

        if hasattr(self, "tsne") and self.tsne:
            self.visualize_tsne(dataset)

    def visualize_tsne(self, dataset):
        X = []  # List for storing the characteristics of the samples
        y = []  # Ready to store the labels of the samples
        # Assuming that 'dataset' is a list or array of your samples
        for idx in range(len(dataset)):
            sample, label = dataset[idx]
            X.append(sample.flatten())
            y.append(label)

        X = np.array(X)
        y = np.array(y)

        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(X)

        plt.figure(figsize=(16, 10))
        sns.scatterplot(
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            hue=y,
            palette=sns.color_palette("hsv", self.num_classes),
            legend="full",
            alpha=0.7,
        )

        plt.title("t-SNE visualization of the dataset")
        plt.xlabel("t-SNE axis 1")
        plt.ylabel("t-SNE axis 2")
        plt.legend(title="Class")
        plt.tight_layout()

        path_to_save_tsne = "./tsne_visualization.png"
        plt.savefig(path_to_save_tsne, dpi=300, bbox_inches="tight")
        plt.close()

    def _adjust_proportions(self, proportions, indices_per_partition, num_samples):
        adjusted = np.array([
            p * (len(indices) < num_samples / self._number)
            for p, indices in zip(proportions, indices_per_partition, strict=False)
        ])
        return adjusted / adjusted.sum()

    def _calculate_class_distribution(self, indices_per_partition, y_data):
        distribution = defaultdict(lambda: np.zeros(self._number))
        for partition_idx, indices in enumerate(indices_per_partition):
            labels, counts = np.unique(y_data[indices], return_counts=True)
            for label, count in zip(labels, counts, strict=False):
                distribution[label][partition_idx] = count
        return {k: v / v.sum() for k, v in distribution.items()}

    @staticmethod
    def _get_targets(dataset) -> np.ndarray:
        if isinstance(dataset.targets, np.ndarray):
            return dataset.targets
        elif hasattr(dataset.targets, "numpy"):
            return dataset.targets.numpy()
        else:
            return np.asarray(dataset.targets)

    def plot_all_data_distribution(self, dataset, partitions_map):
        """

        Plot all of the data distribution of the dataset according to the partitions map provided.

        Args:
            dataset: The dataset to plot (torch.utils.data.Dataset).
            partitions_map: The map of the dataset partitions.
        """
        sns.set()
        sns.set_style("whitegrid", {"axes.grid": False})
        sns.set_context("paper", font_scale=1.5)
        sns.set_palette("Set2")

        num_clients = len(partitions_map)
        num_classes = self.num_classes

        plt.figure(figsize=(12, 8))

        label_distribution = [[] for _ in range(num_classes)]
        for c_id, idc in partitions_map.items():
            for idx in idc:
                label_distribution[dataset.targets[idx]].append(c_id)

        plt.hist(
            label_distribution,
            stacked=True,
            bins=np.arange(-0.5, num_clients + 1.5, 1),
            label=dataset.classes,
            rwidth=0.5,
        )
        plt.xticks(
            np.arange(num_clients),
            ["Participant %d" % (c_id + 1) for c_id in range(num_clients)],
        )
        plt.title("Distribution of splited datasets")
        plt.xlabel("Participant")
        plt.ylabel("Number of samples")
        plt.xticks(range(num_clients), [f" {i}" for i in range(num_clients)])
        plt.legend(loc="upper right")
        plt.tight_layout()

        path_to_save = f"./all_data_distribution_{'iid' if self.is_iid() else 'non_iid'}{'_' + self.get_name()}.png"
        plt.savefig(path_to_save, dpi=300, bbox_inches="tight")
        plt.close()

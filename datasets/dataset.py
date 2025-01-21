from partitions import Partition
import logging
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import defaultdict
import time
import random
import torch
import os
from abc import ABC, abstractmethod
from torch.utils import data


def enable_deterministic():
    seed = 42
    logging.info(f"Fixing randomness with seed {seed}")
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# TODO what is this?
matplotlib.use("Agg")
# TODO what is this?
plt.switch_backend("Agg")


class NebulaDataset(Dataset, ABC):
    """
    Abstract class for a partitioned dataset.

    Classes inheriting from this class need to implement specific methods
    for loading and partitioning the dataset.
    """

    def __init__(
        self,
        partition: Partition,
        num_classes: int,
        batch_size: int,
        num_workers: int,
        seed: int,
    ):
        super().__init__()

        self.partition = partition
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.train_set = None
        self.train_indices_map = None
        self.test_set = None
        self.test_indices_map = None

        # Classes of the participants to be sure that
        # the same classes are used in training and testing
        self.class_distribution = None

        enable_deterministic()

        if partition.get_id() == 0:
            self.initialize_dataset()
        else:
            max_tries = 10
            for i in range(max_tries):
                try:
                    self.initialize_dataset()
                    break
                except Exception as e:
                    logging.info(
                        f"Error loading dataset: {e}. Retrying {i + 1}/{max_tries} in 5 seconds...")
                    time.sleep(5)

    @abstractmethod
    def initialize_dataset(self):
        """
        Initialize the dataset. This should load or create the dataset.
        """
        pass

    # TODO implement this (not abstract)
    @abstractmethod
    def plot(self):
        """
        Plot the partitions of the dataset.
        """
        pass

    def get_train_labels(self):
        """
        Get the labels of the training set based on the indices map.
        """
        if self.train_indices_map is None:
            return None
        return [self.train_set.targets[idx] for idx in self.train_indices_map]

    def get_test_labels(self):
        """
        Get the labels of the test set based on the indices map.
        """
        if self.test_indices_map is None:
            return None
        return [self.test_set.targets[idx] for idx in self.test_indices_map]

    def get_local_test_labels(self):
        """
        Get the labels of the local test set based on the indices map.
        """
        if self.local_test_indices_map is None:
            return None
        return [self.test_set.targets[idx] for idx in self.local_test_indices_map]

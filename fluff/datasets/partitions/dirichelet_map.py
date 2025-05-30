import logging
import numpy as np
from torch.utils import data

from typing import Optional, List

from collections.abc import Sequence

from .partitions import Partition


class DirichletMap(Partition):
    def __init__(
        self,
        partition_id: int,
        partitions_number: int,
        alpha: float = 0.5,
        min_samples_per_class: int = 10,
    ):
        super().__init__(partition_id, partitions_number, alpha)

        self._alpha = alpha
        self._min_samples_per_class = min_samples_per_class

    def get_name(self) -> str:
        return "dirichlet"

    def is_iid(self) -> bool:
        return False

    def generate(self, dataset: data.Dataset, **kwargs):
        """
        Args:
            dataset: the torch.Dataset
            alpha: float,  ratio
            min_samples_per_class: int,
        """

        alpha = self._alpha
        min_samples_per_class = self._min_samples_per_class

        y_data = self._get_targets(dataset)
        unique_labels = np.unique(y_data)
        logging.info(f"Labels unique: {unique_labels}")
        num_samples = len(y_data)

        indices_per_partition: List[List[int]] = [[] for _ in range(self._number)]
        label_distribution = (
            self.class_distribution if self.class_distribution is not None else None
        )

        for label in unique_labels:
            label_indices = np.where(y_data == label)[0]
            np.random.shuffle(label_indices)

            if label_distribution is None:
                proportions = np.random.dirichlet([alpha] * self._number)
            else:
                proportions = label_distribution[label]

            proportions = self._adjust_proportions(
                proportions, indices_per_partition, num_samples
            )
            split_points = (np.cumsum(proportions) * len(label_indices)).astype(int)[
                :-1
            ]

            for partition_idx, indices in enumerate(
                np.split(label_indices, split_points)
            ):
                if len(indices) < min_samples_per_class:
                    indices_per_partition[partition_idx].extend([])
                else:
                    indices_per_partition[partition_idx].extend(indices)

        if label_distribution is None:
            self.class_distribution = self._calculate_class_distribution(
                indices_per_partition, y_data
            )

        return {i: indices for i, indices in enumerate(indices_per_partition)}

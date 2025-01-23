import logging
import numpy as np
from torch.utils import data

from .partitions import Partition


class HomogenousMap(Partition):
    def __init__(self, partition_id: int, partitions_number: int, partition_parameter: float = None):
        super().__init__(partition_id, partitions_number, partition_parameter)

    def get_name(self) -> str:
        return "homogeneous"

    def is_iid(self) -> bool:
        return True

    def generate(self, dataset: data.Dataset, **kwargs):
        assert self.seed, "Seed must be provided"
        assert self.num_classes, "Number of classes must be provided"
        """
        Homogeneously partition the dataset into multiple subsets.

        This function divides a dataset into a specified number of subsets, where each subset
        is intended to have a roughly equal number of samples. This method aims to ensure a
        homogeneous distribution of data across all subsets. It's particularly useful in
        scenarios where a uniform distribution of data is desired among all federated learning
        clients.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to partition. It should have
                                                'data' and 'targets' attributes.

        Returns:
            dict: A dictionary where keys are subset indices (ranging from 0 to partitions_number-1)
                and values are lists of indices corresponding to the samples in each subset.

        The function randomly shuffles the entire dataset and then splits it into the number
        of subsets specified by `partitions_number`. It ensures that each subset has a similar number
        of samples. The function also prints the class distribution in each subset for reference.

        Example usage:
            federated_data = homo_partition(my_dataset)
            # This creates federated data subsets with homogeneous distribution.
        """
        n_nets = self._number

        n_train = len(dataset.targets)
        np.random.seed(self.seed)
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_nets)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_nets)}

        # partitioned_datasets = []
        for i in range(self._number):
            # subset = torch.utils.data.Subset(dataset, net_dataidx_map[i])
            # partitioned_datasets.append(subset)

            # Print class distribution in the current partition
            class_counts = [0] * self.num_classes
            for idx in net_dataidx_map[i]:
                label = dataset.targets[idx]
                class_counts[label] += 1
            logging.info(
                f"Partition {i + 1} class distribution: {class_counts}")

        return net_dataidx_map

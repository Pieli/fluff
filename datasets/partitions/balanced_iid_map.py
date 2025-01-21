import numpy as np

from datasets.partitions import partitions


class BalancedIIDMap(partitions.Partition):
    def __init__(self, partition_id: int, partitions_number: int, partition_parameter: float = None):
        super().__init__(partition_id, partitions_number, partition_parameter)

    def get_name(self) -> str:
        return "balanced iid"

    def is_iid(self) -> bool:
        return True

    def generate(self, dataset, **kwargs):
        """
        Partition the dataset into balanced and IID (Independent and Identically Distributed)
        subsets for each client.

        This function divides a dataset into a specified number of subsets (federated clients),
        where each subset has an equal class distribution. This makes the partition suitable for
        simulating IID data scenarios in federated learning.

        Args:
            dataset (list): The dataset to partition. It should be a list of tuples where each
                            tuple represents a data sample and its corresponding label.

        Returns:
            dict: A dictionary where keys are client IDs (ranging from 0 to partitions_number-1) and
                    values are lists of indices corresponding to the samples assigned to each client.

        The function ensures that each class is represented equally in each subset. The
        partitioning process involves iterating over each class, shuffling the indices of that class,
        and then splitting them equally among the clients. The function does not print the class
        distribution in each subset.

        Example usage:
            federated_data = balanced_iid_partition(my_dataset)
            # This creates federated data subsets with equal class distributions.
        """
        num_clients = self._number
        clients_data = {i: [] for i in range(num_clients)}

        # Get the labels from the dataset
        if isinstance(dataset.targets, np.ndarray):
            labels = dataset.targets
        elif hasattr(dataset.targets, "numpy"):
            # Check if it's a tensor with .numpy() method
            labels = dataset.targets.numpy()
        else:  # If it's a list
            labels = np.asarray(dataset.targets)

        label_counts = np.bincount(labels)
        min_label = label_counts.argmin()
        min_count = label_counts[min_label]

        for label in range(self.num_classes):
            # Get the indices of the same label samples
            label_indices = np.where(labels == label)[0]
            np.random.seed(self.seed)
            np.random.shuffle(label_indices)

            # Split the data based on their labels
            samples_per_client = min_count // num_clients

            for i in range(num_clients):
                start_idx = i * samples_per_client
                end_idx = (i + 1) * samples_per_client
                clients_data[i].extend(label_indices[start_idx:end_idx])

        return clients_data

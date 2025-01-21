import logging
import numpy as np

from datasets.partitions import partitions


class PercentageMap(partitions.Partition):
    def __init__(self, partition_id: int, partitions_number: int, partition_parameter: float = None):
        super().__init__(partition_id, partitions_number, partition_parameter)

    def get_name(self) -> str:
        return "percentage"

    def is_iid(self) -> bool:
        return False

    def generate(self, dataset, **kwargs):
        """
        Partition a dataset into multiple subsets with a specified level of non-IID-ness.

        This function divides a dataset into a specified number of subsets (federated
        clients), where each subset has a different class distribution. The class
        distribution in each subset is determined by a specified percentage, making the
        partition suitable for simulating non-IID (non-Independently and Identically
        Distributed) data scenarios in federated learning.

        Args:
            dataset (torch.utils.data.Dataset): The dataset to partition. It should have
                                                'data' and 'targets' attributes.
            percentage (int): A value between 0 and 100 that specifies the desired
                                level of non-IID-ness for the labels of the federated data.
                                This percentage controls the imbalance in the class distribution
                                across different subsets.

        Returns:
            dict: A dictionary where keys are subset indices (ranging from 0 to partitions_number-1)
                and values are lists of indices corresponding to the samples in each subset.

        The function ensures that the number of classes in each subset varies based on the selected
        percentage. The partitioning process involves iterating over each class, shuffling the
        indices of that class, and then splitting them according to the calculated subset sizes.
        The function also prints the class distribution in each subset for reference.

        Example usage:
            federated_data = percentage_partition(my_dataset, percentage=20)
            # This creates federated data subsets with varying class distributions based on
            # a percentage of 20.
        """
        percentage = kwargs.get("percentage", 20)

        if isinstance(dataset.targets, np.ndarray):
            y_train = dataset.targets
        elif hasattr(dataset.targets, "numpy"):  # Check if it's a tensor with .numpy() method
            y_train = dataset.targets.numpy()
        else:  # If it's a list
            y_train = np.asarray(dataset.targets)

        num_classes = self.num_classes
        num_subsets = self._number
        class_indices = {i: np.where(y_train == i)[
            0] for i in range(num_classes)}

        # Get the labels from the dataset
        labels = np.array([dataset.targets[idx]
                          for idx in range(len(dataset))])
        label_counts = np.bincount(labels)

        min_label = label_counts.argmin()
        min_count = label_counts[min_label]

        classes_per_subset = int(num_classes * percentage / 100)
        if classes_per_subset < 1:
            raise ValueError(
                "The percentage is too low to assign at least one class to each subset.")

        subset_indices = [[] for _ in range(num_subsets)]
        class_list = list(range(num_classes))
        np.random.seed(self.seed)
        np.random.shuffle(class_list)

        for i in range(num_subsets):
            for j in range(classes_per_subset):
                # Use modulo operation to cycle through the class_list
                class_idx = class_list[(
                    i * classes_per_subset + j) % num_classes]
                indices = class_indices[class_idx]
                np.random.seed(self.seed)
                np.random.shuffle(indices)
                # Select approximately 50% of the indices
                subset_indices[i].extend(indices[: min_count // 2])

            class_counts = np.bincount(
                np.array([dataset.targets[idx] for idx in subset_indices[i]]))
            logging.info(
                f"Partition {i + 1} class distribution: {class_counts.tolist()}")

        partitioned_datasets = {
            i: subset_indices[i] for i in range(num_subsets)}

        return partitioned_datasets

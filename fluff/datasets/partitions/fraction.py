import numpy as np

from .partitions import Partition


class BalancedFraction(Partition):
    def __init__(self, percent: float = 0.5):
        super().__init__(partition_id=0, partitions_number=1, partition_parameter=None)
        self._percent = percent
        assert 0 < self._percent < 1, "Fraction should be between 0 and 1"

    def get_name(self) -> str:
        return "balanced iid"

    def is_iid(self) -> bool:
        return True

    def generate(self, dataset, **kwargs):
        clients_data = {i: [] for i in range(2)}

        # Get the labels from the dataset
        if isinstance(dataset.targets, np.ndarray):
            labels = dataset.targets
        elif hasattr(dataset.targets, "numpy"):
            # Check if it's a tensor with .numpy() method
            labels = dataset.targets.numpy()
        else:  # If it's a list
            labels = np.asarray(dataset.targets)

        for label in range(self.num_classes):
            # Get the indices of the same label samples
            label_indices = np.where(labels == label)[0]
            np.random.seed(self.seed)
            np.random.shuffle(label_indices)

            end_idx = round(len(label_indices) * self._percent)
            clients_data[0].extend(label_indices[:end_idx])
            clients_data[1].extend(label_indices[end_idx:])

        return clients_data

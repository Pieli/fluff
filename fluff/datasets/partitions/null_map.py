from torch.utils import data

from .partitions import Partition


class NullMap(Partition):
    def __init__(self, partition_id: int, partitions_number: int, partition_parameter: float = None):
        super().__init__(partition_id, partitions_number, partition_parameter)

    def get_name(self) -> str:
        return "nullmap"

    def is_iid(self) -> bool:
        return False

    def generate(self, dataset: data.Dataset, **kwargs):
        """
        Args:
            dataset: the torch.Dataset
            alpha: float,  ratio
            min_samples_per_class: int,
        """
        return {i: list(range(len(dataset))) for i in range(self._number)}

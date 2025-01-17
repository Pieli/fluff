from abc import ABC, abstractmethod
from torch.utils import data


class Dset(data.Dataset, ABC):
    def __init__(self, dataset):
        pass

    @abstractmethod
    def load(self):
        pass

    def plot(self):
        pass


# balanced
# unbalanced
# dataset = BalancedIID(Dataset()).load()

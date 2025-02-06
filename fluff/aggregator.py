import torch
import copy
from abc import ABC, abstractmethod
from typing import List, Any, Dict
import lightning as pl


class Aggregator(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def run(self, models: List[Any]):
        pass


class Noop(Aggregator):
    def __init__(self) -> None:
        super().__init__()

    def run(self, models: List[Any]):
        print(f"Got {len(models)} models: {models}")


class FedAvg(Aggregator):
    def __init__(self) -> None:
        super().__init__()

    def run(self, models: List[pl.LightningModule]) -> Dict[str, Any]:
        m_states = [copy.deepcopy(m.state_dict()) for m in models]

        # Check that the list is not empty
        if not m_states:
            raise ValueError("The models list should not be empty.")

        # Create a new model to store the average
        avg_state = copy.deepcopy(m_states[0])

        with torch.no_grad():
            # Go over each parameter in the model
            for param_name in avg_state:
                # Accumulate and average the parameter values across all state_dicts
                avg_state[param_name] = sum(
                    state[param_name] for state in m_states
                ) / len(m_states)

        return avg_state

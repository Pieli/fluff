from argparse import Namespace
from typing import Any, Optional, Union
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.fabric.utilities.types import _PATH

from typing_extensions import override
from collections.abc import Mapping

import logging


class FluffTensorBoardLogger(TensorBoardLogger):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):

        super().__init__(*args, **kwargs)
        self.local_step = 0
        self.global_step = 0

    @override
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        print(metrics, step)
        if step is None:
            self.local_step += 1
            step = self.global_step + self.local_step

        # if "epoch" in metrics:
            # metrics.pop("epoch")

        return super().log_metrics(metrics=metrics, step=step)

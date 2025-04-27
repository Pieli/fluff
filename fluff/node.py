import torch
from torch.utils import data
import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

from typing import Union, Optional, Any, Mapping

from .datasets.dataset import NebulaDataset

from fluff import logger
from argparse import Namespace


class Node:
    def __init__(
        self,
        name: str,
        experiement_name: str,
        model: pl.LightningModule,
        dataset: NebulaDataset,
        num_workers: int = 2,
        seed: Optional[int] = None,
        hp: Optional[Namespace] = None,
        logger_cls: TensorBoardLogger = None,
    ) -> None:

        self._name = name
        self._model = model
        self._dataset = dataset
        self._num_workers = num_workers
        self._seed = seed

        self._test_trainer: Optional[pl.Trainer] = None

        logger_class = logger_cls or TensorBoardLogger
        self._logger = logger_class(
            f"./fluff_logs/{experiement_name}",
            self._name,
            type(self._model).__name__,
        )

        if hp:
            self._logger.log_hyperparams(params=hp)

    def setup(self):
        training_dataset = data.Subset(
            self._dataset.train_set, self._dataset.train_indices_map
        )
        test_dataset = self._dataset.test_set

        train_set_size = int(len(training_dataset) * 0.8)
        valid_set_size = len(training_dataset) - train_set_size

        # split the train set into two
        seed = torch.Generator().manual_seed(42)
        train_set, valid_set = data.random_split(
            training_dataset, [train_set_size, valid_set_size], generator=seed
        )

        generator = torch.Generator().manual_seed(self._seed) if self._seed else None
        self.train_loader = data.DataLoader(
            train_set,
            batch_size=self._dataset.get_batch_size(),
            shuffle=True,
            num_workers=self._num_workers,
            generator=generator,
        )

        self.val_loader = data.DataLoader(
            valid_set,
            batch_size=self._dataset.get_batch_size(),
            shuffle=False,
            num_workers=self._num_workers,
        )

        self.test_loader = data.DataLoader(
            test_dataset,
            batch_size=self._dataset.get_batch_size(),
            shuffle=False,
            num_workers=self._num_workers,
        )

        return self

    def tune(self, min_lr: int, max_lr: int, num_training: int) -> None:

        import matplotlib
        from lightning.pytorch.tuner import Tuner

        try:
            matplotlib.use("QtAgg")
        except:
            pass

        trainer = pl.Trainer()

        tuner = Tuner(trainer)

        lr_finder = tuner.lr_find(
            model=self._model,
            train_dataloaders=self.train_loader,
            mode="linear",
            min_lr=min_lr,
            max_lr=max_lr,
            num_training=num_training,
        )

        print(f"[->] Suggested value {lr_finder.suggestion()}")  # type: ignore

        return lr_finder.results  # type: ignore

    def train(
        self,
        epochs: int,
        dev_runs: Union[bool | int] = False,
        skip_val=False,
        swap_val_to_test=True,
        skip_test=False,
        callbacks=None,
        ckpt_path=None,
        strat: Any = "auto",
        **kwargs,
    ) -> None:

        enable_prog = (
            kprog
            if (kprog := kwargs.get("enable_progress_bar")) is not None
            else not isinstance(
                self._logger,
                logger.FluffTensorBoardLogger,
            )
        )

        kwargs.pop("enable_progress_bar", None)

        trainer = pl.Trainer(
            max_epochs=epochs,
            fast_dev_run=dev_runs,
            logger=self._logger,
            deterministic=True,
            enable_progress_bar=enable_prog,
            enable_checkpointing=True,
            callbacks=callbacks,
            strategy=strat,
            accelerator="gpu",
            devices=1,
            **kwargs,
        )

        val = self.val_loader if not skip_val else None
        val = self.test_loader if swap_val_to_test else val

        if swap_val_to_test:
            print("[!]   Using Test set as the validation set")

        trainer.fit(
            model=self._model,
            train_dataloaders=self.train_loader,
            val_dataloaders=val,
            ckpt_path=ckpt_path,
        )

        if not skip_test:
            trainer.test(model=self._model, dataloaders=self.test_loader)

    def test(self, epochs=10, keep_trainer=True) -> list[Mapping[str, float]]:
        if not self._test_trainer:
            self._test_trainer = pl.Trainer(
                max_epochs=epochs,
                logger=self._logger,
            )

        output = self._test_trainer.test(
            model=self._model, dataloaders=self.test_loader
        )

        self._logger.experiment.add_scalar(
            "server/test_f1", output[0]["test_f1"], global_step=epochs
        )

        return output

    def get_model(self) -> pl.LightningModule:
        return self._model

    def get_name(self) -> str:
        return self._name

    def __repr__(self) -> str:

        model_lines = [13 * " " + line for line in repr(self._model).splitlines()]
        model_lines[0] = model_lines[0].lstrip()

        l4 = " " * 4
        model_lines_str = "\n".join(model_lines)
        return "".join(
            (
                "Node(\n",
                f"{l4}(name): {repr(self._name)}\n",
                f"{l4}(model): {model_lines_str}\n",
                f"{l4}(dataset): {repr(self._dataset)}\n",
                f"{l4}(num_workers): {self._num_workers}\n)",
            )
        )

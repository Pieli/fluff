import torch
import torchmetrics
import lightning as pl
from torch import nn
import torch.nn.functional as F

import utils

from typing import Sequence


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# architecture of nebula platform (for comparing)
class CNNv2(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 512)
        self.fc2 = torch.nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LitCNN(pl.LightningModule):
    def __init__(self, model: nn.Module, num_classes: int = 10, lr: float = 0.01):
        super().__init__()

        self.lr = lr

        assert isinstance(model, nn.Module)

        self.cnn = model
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.test_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        self._recorded_statistics: torch.Tensor = torch.zeros(10)

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self._recorded_statistics += torch.bincount(y, minlength=10).cpu()

        loss = self.criterion(y_hat, y)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y)
        self.test_acc(y_hat, y)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=3e-4)

    def get_statistics(self) -> torch.tensor:
        return self._recorded_statistics


class ServerLitCNNCifar100(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        ensemble: list[nn.Module],
        distillation_phase: bool = False,
    ):
        super().__init__()

        assert isinstance(model, nn.Module)

        self.cnn = model
        self.ensemble = nn.ModuleList(ensemble)
        self.criterion = nn.CrossEntropyLoss()
        self._distillation_phase = distillation_phase
        self._count_stats: Sequence[torch.Tensor] = tuple(torch.empty(1))

        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=10
        )
        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=10
        )
        self.test_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=10
        )

        self.train_div = torchmetrics.KLDivergence(log_prob=True, reduction="mean")

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        batch_logits = []
        for ens in self.ensemble:
            with torch.no_grad():
                ens_y_hat = ens.forward(x)

            batch_logits.append(ens_y_hat)

        ens_logits = utils.alternative_avg(
            batch_logits, self._count_stats, 10, len(self.ensemble)
        )

        # loss = self.kl_divergence(y_hat, ens_logits)
        loss = self.l2_distillation(y_hat, ens_logits)

        self.train_div(y_hat, ens_logits)
        self.train_acc(y_hat.argmax(dim=1), ens_logits.argmax(dim=1))
        self.log("train_kl_div", self.train_div, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    # TODO test this
    def l2_distillation(
        self, server_log: torch.Tensor, ensemble_log: torch.Tensor
    ) -> torch.Tensor:
        return F.mse_loss(
            torch.sigmoid(server_log), torch.sigmoid(ensemble_log), reduction="sum"
        ) * (1 / server_log.size(1))

    def kl_divergence(self, p, q):
        return (
            F.kl_div(
                F.log_softmax(p / 3, dim=1),
                F.softmax(q / 3, dim=1),
                reduction="batchmean",
            )
            * 9
        )

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)

        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y)

        self.test_acc(y_hat, y)
        self.log("test_acc", self.test_acc, on_step=True, on_epoch=True)
        self.log("test_loss", test_loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.cnn.parameters(), lr=1e-3)

    def set_distillation_phase(self, distillation: bool):
        assert isinstance(distillation, bool)
        self._distillation_phase = distillation
        return self

    def set_count_statistics(self, counts: list[torch.Tensor]):
        self._count_stats = tuple(stat.cuda() for stat in counts)
        return self

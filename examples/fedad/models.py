import torch
import torchmetrics
import lightning as pl
from torch import nn
import torch.nn.functional as F

import utils


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


class LitCNN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.cnn = CNN(num_classes=10)
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=10)
        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=10)

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)

        self.val_acc(y_hat, y)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        # TODO do cosine annealing
        return torch.optim.SGD(self.parameters(), lr=0.0025, weight_decay=3e-4)


class LitCNN_Cifar100(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()

        assert isinstance(model, nn.Module)

        self.cnn = model
        self.criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=100)
        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=100)

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0025, weight_decay=3e-4)


class ServerLitCNNCifar100(pl.LightningModule):
    def __init__(self, model: nn.Module, distillation_phase: bool = False, ensemble: list[nn.Module] = []):
        super().__init__()

        assert isinstance(model, nn.Module)

        self.cnn = model
        self.ensemble = nn.ModuleList(ensemble)
        self.criterion = nn.CrossEntropyLoss()
        self._distillation_phase = distillation_phase

        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=100)
        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=100)

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        if self._distillation_phase:
            (serv_result, serv_counts) = utils.average_logits_per_class(
                y_hat, y, 100,
            )

            batch_logits = []
            batch_counts = []
            for ens in self.ensemble:
                ens_y_hat = ens.forward(x)
                (result, counts) = utils.average_logits_per_class(
                    ens_y_hat, y, 100,
                )

                batch_logits.append(result)
                batch_counts.append(counts)

            ens_logits = utils.logits_ensemble_eq_3(batch_logits,
                                                    batch_counts,
                                                    100,
                                                    len(self.ensemble))

            loss = self.l2_distillation(result, ens_logits)
        else:
            loss = self.criterion(y_hat, y)

        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    # TODO test this
    def l2_distillation(self, server_log: torch.Tensor, ensemble_log: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(server_log, ensemble_log, reduction="sum") * (1 / server_log.size(1))

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        if self._distillation_phase:
            return torch.optim.Adam(self.cnn.parameters(), lr=1e-3, weight_decay=0)
        return torch.optim.SGD(self.cnn.parameters(), lr=0.0025, weight_decay=3e-4)

    def set_distillation_phase(self, distillation: bool):
        assert isinstance(distillation, bool)
        self._distillation_phase = distillation
        return self

import torch
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

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        # TODO do cosine annealing
        return torch.optim.SGD(self.parameters(), lr=0.0025, weight_decay=3e-4)


class LitCNN_Cifar100(pl.LightningModule):
    def __init__(self, model: nn.Module, distillation: bool = False):
        super().__init__()

        assert isinstance(model, nn.Module)

        self.cnn = model
        self._distillation = distillation
        self.criterion = nn.CrossEntropyLoss()

        self._recorded_counts = []
        self._recorded_logits = []

        self.epoch_logits = []
        self.epoch_counts = []

        self._average_logits = None
        self._class_counts = None

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        (result, counts) = utils.average_logits_per_class(y_hat.detach(), y, 100)

        self._recorded_logits.append(result)
        self._recorded_counts.append(counts)

        loss = self.criterion(y_hat, y)
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y)
        self.log("test_loss", test_loss)

    def on_train_epoch_end(self):
        if self._distillation:
            self._recorded_logits = []
            self._recorded_counts = []
            return

        if len(self._recorded_logits) <= 0:
            return

        print("Distillation on Epoch end")

        logits = utils.logits_ensemble_eq_3(self._recorded_logits,
                                            self._recorded_counts,
                                            100,
                                            len(self._recorded_logits))

        self.epoch_logits.append(logits)
        self.epoch_counts.append(torch.stack(self._recorded_counts).sum(dim=0))

        self._recorded_logits = []
        self._recorded_counts = []

    def on_train_end(self):
        if self._distillation:
            return

        print("Training end")

        if len(self.epoch_logits) <= 0:
            return

        self._average_logits = utils.logits_ensemble_eq_3(self.epoch_logits,
                                                          self.epoch_counts,
                                                          100,
                                                          len(self.epoch_logits),
                                                          )

        self._class_counts = torch.stack(self.epoch_counts).sum(dim=0)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0025, weight_decay=3e-4)

    def get_average_logits(self) -> torch.Tensor:
        return self._average_logits

    def get_class_counts(self) -> torch.Tensor:
        return self._class_counts

    def set_distillation(self, distillation: bool) -> None:
        assert isinstance(distillation, bool)
        self._distillation = distillation

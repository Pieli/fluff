import torch
import lightning as pl
from torch import nn
import torch.nn.functional as F

import utils


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, 10)

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

        self.cnn = CNN()
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
    def __init__(self, model: nn.Module):
        super().__init__()

        self.cnn = model
        self.criterion = nn.CrossEntropyLoss()

        self.class_counts = []
        self.recorded_logits = []

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        (result, counts) = utils.average_logits_per_class(y_hat, y, 100)
        self.recorded_logits.append(result)
        self.class_counts.append(counts)

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

import time
import torch
import torchmetrics
import lightning as pl
from torch import nn
import torch.nn.functional as F

import utils

from typing import Sequence

from gradcam import GradCAM


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
        distillation: str,
    ):
        super().__init__()

        assert isinstance(model, nn.Module)
        assert distillation in ("kl", "l2", "l2_new")

        self.cnn = model
        self.ensemble = nn.ModuleList(ensemble)
        self.criterion = nn.CrossEntropyLoss()

        # logits distillation
        if distillation == "kl":
            self.dist_criterion = self.kl_divergence
        elif distillation == "l2_new":
            self.dist_criterion = self.l2_distillation_new  # type: ignore
        elif distillation == "l2":
            self.dist_criterion = self.l2_distillation  # type: ignore
        else:
            raise Exception("Some input is wrong")

        # gradcam
        self.ensemble_cams = [GradCAM(mod, "layer3.2.conv2") for mod in self.ensemble]
        self.server_cam = GradCAM(self.cnn, "layer3.2.conv2")

        # statistics
        self._count_stats: Sequence[torch.Tensor] = tuple(torch.empty(1))

        # metrics
        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=10
        )
        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=10
        )
        self.test_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=10
        )

        self.train_div = torchmetrics.KLDivergence(
            log_prob=False,
            reduction="mean",
        )

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        batch_logits = [ens.forward(x) for ens in self.ensemble]

        ens_logits = utils.alternative_avg(
            raw_logits=batch_logits,
            raw_statistics=self._count_stats,
            num_classes=10,
            num_nodes=len(self.ensemble),
        )

        loss = self.dist_criterion(y_hat, ens_logits, T=3)

        cam_generation_start = time.time()
        class_cams, server_cams = self.cam_generation(
            batch_logits=batch_logits,
            server_logits=y_hat,
            num_samples=1,
            top=2,
        )

        del batch_logits

        # print(f"#> cam_generation {(time.time() - cam_generation_start):.4f}s")

        union_loss = self.union_loss(server_cams, class_cams)
        inter_loss = self.inter_loss(server_cams, class_cams)


        self.train_div(torch.softmax(y_hat, dim=1), torch.softmax(ens_logits, dim=1))
        self.train_acc(y_hat.argmax(dim=1), ens_logits.argmax(dim=1))
        self.log("train_kl_div", self.train_div, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        # self.log("train_union_loss", union_loss, on_step=True)
        # self.log("train_inter_loss", inter_loss, on_step=True)

        return loss

    def union_loss(self, server_cams, client_cams):
        loss = utils.loss_union(
            client_cams.amax(dim=(1, 2)),
            server_cams.amax(dim=(1,)),
            num_classes=10,
        )

        return loss

    def inter_loss(self, server_cams, client_cams):
        loss = utils.loss_intersection(
            client_cams.amin(dim=(1, 2)),
            server_cams.amin(dim=(1,)),
            num_classes=10,
        )

        return loss

    def cam_generation(self, batch_logits, server_logits, num_samples, top):
        assert len(batch_logits) > 0

        classes = 10

        weights = utils.node_weights(
            node_stats=torch.stack(self._count_stats),
            num_classes=classes,
            num_nodes=len(self.ensemble),
        )

        # sampled_nodes = [[0, 1, 2]]
        sampled_nodes = utils.sample_with_top(
            weights,
            num_out_samples=num_samples,
            top=top,
        )

        batch_size = batch_logits[0].size(1)
        device = batch_logits[0].device

        class_maps = []
        server_maps = []
        for c_ind, selected in enumerate(sampled_nodes):
            target = torch.full((batch_size,), c_ind, dtype=torch.int, device=device)

            start_time = time.time()

            node_maps = [
                self.ensemble_cams[node_ind].generate_from_logits(
                    batch_logits[node_ind],
                    target,
                )
                for node_ind in selected
            ]

            # print(f"--- {(time.time() - start_time):.4f} seconds ---")

            server_cam = self.server_cam.generate_from_logits(
                server_logits,
                target,
            )

            server_maps.append(server_cam)
            class_maps.append(torch.stack(node_maps))

        return torch.stack(class_maps), torch.stack(server_maps)

    def l2_distillation(
        self,
        server_log: torch.Tensor,
        ensemble_log: torch.Tensor,
        T: int = 3,
    ) -> torch.Tensor:

        return F.mse_loss(
            torch.sigmoid(server_log),
            torch.sigmoid(ensemble_log),
            reduction="mean",
        )

    def l2_distillation_new(
        self,
        server_log: torch.Tensor,
        ensemble_log: torch.Tensor,
        T: int = 3,
    ) -> torch.Tensor:

        return (
            torch.linalg.vector_norm(
                server_log - ensemble_log,
                ord=2,
            )
            / 10
        )

    def kl_divergence(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        T: int = 3,
    ) -> torch.Tensor:

        return (
            F.kl_div(
                F.log_softmax(p / T, dim=1),
                F.softmax(q / T, dim=1),
                reduction="batchmean",
            )
            * T
            * T
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

    def set_count_statistics(self, counts: list[torch.Tensor]):
        self._count_stats = tuple(stat.cuda() for stat in counts)
        return self

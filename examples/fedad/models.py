import torch
import torchmetrics
from torch import nn

import utils

from typing import Sequence

from gradcam import GradCAM

from fluff.models import LitModel


class LitCNN(LitModel):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        lr: float = 0.01,
    ):
        super().__init__(model, num_classes, lr)

        assert isinstance(model, nn.Module)

        self.lr = lr
        self.cnn = model
        self.criterion = nn.CrossEntropyLoss()
        self._recorded_statistics: torch.Tensor = torch.zeros(10)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self._recorded_statistics += torch.bincount(y, minlength=10).cpu()

        loss = self.criterion(y_hat, y)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=3e-4)

    def get_statistics(self) -> torch.tensor:
        return self._recorded_statistics


class ServerLitCNNCifar100(LitModel):
    def __init__(
        self,
        model: nn.Module,
        ensemble: list[nn.Module],
        distillation: str,
    ):
        super().__init__(model, num_classes=10, lr=1e-3)

        assert isinstance(model, nn.Module)
        assert distillation in ("kl", "l2", "l2_new")

        self.cnn = model
        self.ensemble = nn.ModuleList(ensemble)
        self.criterion = nn.CrossEntropyLoss()

        # logits distillation
        if distillation == "kl":
            self.dist_criterion = utils.kl_divergence
        elif distillation == "l2_new":
            self.dist_criterion = utils.l2_distillation_new  # type: ignore
        elif distillation == "l2":
            self.dist_criterion = utils.l2_distillation  # type: ignore
        else:
            raise Exception("Some input is wrong")

        # gradcam
        self.ensemble_cams = [GradCAM(mod, "layer3.2.conv2") for mod in self.ensemble]
        self.server_cam = GradCAM(self.cnn, "layer3.2.conv2")

        # statistics
        self._count_stats: Sequence[torch.Tensor] = tuple(torch.empty(1))

        # metrics
        self.train_div = torchmetrics.KLDivergence(
            log_prob=False,
            reduction="mean",
        )

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

        logits_loss = self.dist_criterion(y_hat, ens_logits, T=3)

        # cam_generation_start = time.time()
        class_cams, server_cams = self.cam_generation(
            batch_logits=batch_logits,
            server_logits=y_hat,
            num_samples=1,
            top=1,
        )

        # print(f"#> cam_generation {(time.time() - cam_generation_start):.4f}s")

        union_loss = utils.loss_union(
            class_cams.amax(dim=(1,)),  # reduce nodes dimension
            server_cams,
            num_classes=10,
        )

        inter_loss = utils.loss_intersection(
            class_cams.amin(dim=(1,)),  # reduce nodes dimension
            server_cams,
            num_classes=10,
        )

        total_loss = logits_loss + union_loss + inter_loss

        self.train_div(torch.softmax(y_hat, dim=1), torch.softmax(ens_logits, dim=1))
        self.train_acc(y_hat.argmax(dim=1), ens_logits.argmax(dim=1))
        self.log("train_kl_div", self.train_div, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_logits_loss", logits_loss, on_step=False, on_epoch=True)
        self.log("train_union_loss", union_loss, on_step=True, on_epoch=True)
        self.log("train_inter_loss", inter_loss, on_step=True, on_epoch=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True)

        return total_loss

    def cam_generation(self, batch_logits, server_logits, num_samples, top):
        assert len(batch_logits) > 0

        weights = utils.node_weights(
            node_stats=torch.stack(self._count_stats),
            num_classes=10,
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

            # start_time = time.time()
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

    def configure_optimizers(self):
        return torch.optim.Adam(self.cnn.parameters(), lr=1e-3)

    def set_count_statistics(self, counts: list[torch.Tensor]):
        self._count_stats = tuple(stat.cuda() for stat in counts)
        return self

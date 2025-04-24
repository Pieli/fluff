import copy
import torch
import torchmetrics
from torch import nn
from torch.nn.functional import cosine_similarity

import utils
from typing import Sequence, Optional, OrderedDict

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
        self.train_f1(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr, weight_decay=3e-4)

    def get_statistics(self) -> torch.tensor:
        return self._recorded_statistics


class FedProxModel(LitModel):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        lr: float = 0.01,
        mu: float = 0.5,
    ):
        super().__init__(model, num_classes, lr)

        assert isinstance(model, nn.Module)

        self.mu = mu
        self.lr = lr
        self.cnn = model
        self.criterion = nn.CrossEntropyLoss()

        self.global_model = None

    def on_train_start(self):
        self.global_model = copy.deepcopy(self.cnn)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        proximal_term = 0.0
        for w, w_t in zip(
            self.cnn.parameters(),
            self.global_model.parameters(),
        ):
            proximal_term += (w - w_t).norm(2)

        loss = self.criterion(y_hat, y) + (self.mu / 2) * proximal_term

        self.train_acc(y_hat, y)
        self.train_f1(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.cnn.parameters(), lr=self.lr, weight_decay=3e-4)


class MoonModel(LitModel):
    def __init__(
        self,
        model: nn.Module,
        num_classes: int = 10,
        lr: float = 0.01,
        mu: float = 0.01,
        tau: float = 0.3,
    ):
        super().__init__(model, num_classes, lr)

        assert isinstance(model, nn.Module)

        self.tau = tau
        self.mu = mu
        self.lr = lr
        self.cnn = model
        self.criterion = nn.CrossEntropyLoss()

        self.global_model: nn.Module = copy.deepcopy(self.cnn)
        self.prev_model: nn.Module = copy.deepcopy(self.cnn)

    def set_global_model(self, global_state: OrderedDict):
        self.global_model.load_state_dict(global_state)

    def on_train_start(self):
        self.prev_model = copy.deepcopy(self.cnn)

    def training_step(self, batch, batch_idx):
        x, y = batch

        z_curr = self.cnn.get_last_features(x, detach=False)
        z_global = self.global_model.get_last_features(x, detach=True)
        z_prev = self.prev_model.get_last_features(x, detach=True)
        logits = self.cnn.classifier(z_curr)

        loss_sup = self.criterion(logits, y)
        loss_con = -torch.log(
            torch.exp(
                cosine_similarity(
                    z_curr.flatten(1),
                    z_global.flatten(1),
                )
                / self.tau
            )
            / (
                torch.exp(
                    cosine_similarity(
                        z_prev.flatten(1),
                        z_curr.flatten(1),
                    )
                    / self.tau
                )
                + torch.exp(
                    cosine_similarity(
                        z_curr.flatten(1),
                        z_global.flatten(1),
                    )
                    / self.tau
                )
            )
        )

        loss = loss_sup + self.mu * torch.mean(loss_con)

        self.train_acc(logits, y)
        self.train_f1(logits, y)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.cnn.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-5
        )


class ServerLitCNNCifar100(LitModel):
    def __init__(
        self,
        model: nn.Module,
        ensemble: list[nn.Module],
        distillation: str,
        lr: float = 1e-3,
    ):
        super().__init__(model, num_classes=10, lr=lr)

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

        amax_hat = y_hat.argmax(dim=1)
        amax_ens = ens_logits.argmax(dim=1)
        self.train_acc(amax_hat, amax_ens)
        self.train_f1(amax_hat, amax_ens)
        self.log("train_kl_div", self.train_div, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True)
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
        return torch.optim.Adam(self.cnn.parameters(), lr=self.lr)

    def set_count_statistics(self, counts: list[torch.Tensor]):
        self._count_stats = tuple(stat.cuda() for stat in counts)
        return self


class ServerLitCifar100LogitsOnly(ServerLitCNNCifar100):
    def __init__(
        self,
        model: nn.Module,
        ensemble: list[nn.Module],
        distillation: str,
        lr: float = 1e-3,
    ):
        super().__init__(model, ensemble, distillation, lr)
        del self.ensemble_cams
        del self.server_cam

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        with torch.no_grad():
            batch_logits = [ens.forward(x) for ens in self.ensemble]

        ens_logits = utils.alternative_avg(
            raw_logits=batch_logits,
            raw_statistics=self._count_stats,
            num_classes=10,
            num_nodes=len(self.ensemble),
        )

        logits_loss = self.dist_criterion(y_hat, ens_logits, T=3)

        total_loss = logits_loss

        amax_hat = y_hat.argmax(dim=1)
        amax_ens = ens_logits.argmax(dim=1)
        self.train_div(torch.softmax(y_hat, dim=1), torch.softmax(ens_logits, dim=1))
        self.train_acc(amax_hat, amax_ens)
        self.train_f1(amax_hat, amax_ens)
        self.log("train_kl_div", self.train_div, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True)
        self.log("train_logits_loss", logits_loss, on_step=False, on_epoch=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True)

        return total_loss


class ServerCifar10CEandLogits(ServerLitCNNCifar100):
    def __init__(
        self,
        model: nn.Module,
        ensemble: list[nn.Module],
        distillation: str,
        lr: float = 1e-3,
    ):
        super().__init__(model, ensemble, distillation, lr)
        del self.ensemble_cams
        del self.server_cam

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        with torch.no_grad():
            batch_logits = [ens.forward(x) for ens in self.ensemble]

        ens_logits = utils.alternative_avg(
            raw_logits=batch_logits,
            raw_statistics=self._count_stats,
            num_classes=10,
            num_nodes=len(self.ensemble),
        )

        ce_loss = self.criterion(y_hat, y)
        logits_loss = self.dist_criterion(y_hat, ens_logits, T=3)

        total_loss = ce_loss + logits_loss

        amax_hat = y_hat.argmax(dim=1)
        amax_ens = ens_logits.argmax(dim=1)
        self.train_div(torch.softmax(y_hat, dim=1), torch.softmax(ens_logits, dim=1))
        self.train_acc(amax_hat, amax_ens)
        self.train_f1(amax_hat, amax_ens)
        self.log("train_kl_div", self.train_div, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True)
        self.log("train_logits_loss", logits_loss, on_step=False, on_epoch=True)
        self.log("train_ce_loss", ce_loss, on_step=False, on_epoch=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True)

        return total_loss


class LMDServerLitCifar100LogitsOnly(ServerLitCNNCifar100):
    def __init__(
        self,
        model: nn.Module,
        ensemble: list[nn.Module],
        distillation: str,
        counts: torch.Tensor,
        lr: float = 1e-3,
    ):
        super().__init__(model, ensemble, distillation, lr)
        del self.ensemble_cams
        del self.server_cam

        n_sum = sum(counts)
        self.majority = torch.tensor(
            [i for i, n in enumerate(counts) if n > 0 and n >= (n_sum / n)]
        )
        print(self.majority)

        weights = counts / n_sum
        print("weights", weights)
        # self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.criterion = nn.CrossEntropyLoss()

    def lmd_divergence(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        gold: torch.Tensor,
        majority: torch.Tensor,
        T: int = 3,
    ) -> torch.Tensor:

        return (
            nn.functional.kl_div(
                utils.log_softmax_mod(input / T, gold, dim=1),
                utils.softmax_mod(target / T, majority, dim=1),
                reduction="batchmean",
            )
            * T
            * T
        )

    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x)

        with torch.no_grad():
            batch_logits = [ens.forward(x) for ens in self.ensemble]

        ens_logits = utils.alternative_avg(
            raw_logits=batch_logits,
            raw_statistics=self._count_stats,
            num_classes=10,
            num_nodes=len(self.ensemble),
        )

        logits_loss = self.lmd_divergence(
            input=y_hat,
            target=ens_logits,
            gold=y,
            majority=self.majority,
            T=1,
        )

        # logits_loss = self.dist_criterion(y_hat, ens_logits, T=1)

        ce = self.criterion(y_hat, y)
        total_loss = logits_loss + ce
        # total_loss = ce

        amax_hat = y_hat.argmax(dim=1)
        self.train_div(torch.softmax(y_hat, dim=1), torch.softmax(ens_logits, dim=1))
        self.train_acc(amax_hat, y)
        self.train_f1(amax_hat, y)
        self.log("train_kl_div", self.train_div, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True)
        self.log("train_logits_loss", logits_loss, on_step=False, on_epoch=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True)
        return total_loss

    """
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y)

        self.val_acc(y_hat, y)
        self.val_f1(y_hat, y)

        class_f1 = self.val_f1_class(y_hat, y)
        metrics = {f"classes/f1_class_{i}": f1 for i, f1 in enumerate(class_f1)}
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=True,
        )
        self.log("val_acc", self.val_acc, on_step=True, on_epoch=True)
        self.log("val_f1", self.val_f1, on_step=True, on_epoch=True)
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
    """


class ServerLitCifar100InterOnly(ServerLitCNNCifar100):
    def __init__(
        self,
        model: nn.Module,
        ensemble: list[nn.Module],
        distillation: str,
        lr: float = 1e-3,
    ):
        super().__init__(model, ensemble, distillation, lr)

    def training_step(self, batch, batch_idx):
        x, _ = batch
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

        inter_loss = utils.loss_intersection(
            class_cams.amin(dim=(1,)),  # reduce nodes dimension
            server_cams,
            num_classes=10,
        )

        total_loss = logits_loss + inter_loss

        amax_hat = y_hat.argmax(dim=1)
        amax_ens = ens_logits.argmax(dim=1)
        self.train_div(torch.softmax(y_hat, dim=1), torch.softmax(ens_logits, dim=1))
        self.train_acc(amax_hat, amax_ens)
        self.train_f1(amax_hat, amax_ens)
        self.log("train_kl_div", self.train_div, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True)
        self.log("train_logits_loss", logits_loss, on_step=False, on_epoch=True)
        self.log("train_inter_loss", inter_loss, on_step=True, on_epoch=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True)

        return total_loss


class ServerLitCifar100UnionOnly(ServerLitCNNCifar100):
    def __init__(
        self,
        model: nn.Module,
        ensemble: list[nn.Module],
        distillation: str,
        lr: float = 1e-3,
    ):
        super().__init__(model, ensemble, distillation, lr)

    def training_step(self, batch, batch_idx):
        x, _ = batch
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

        union_loss = utils.loss_union(
            class_cams.amax(dim=(1,)),  # reduce nodes dimension
            server_cams,
            num_classes=10,
        )

        total_loss = logits_loss + union_loss

        amax_hat = y_hat.argmax(dim=1)
        amax_ens = ens_logits.argmax(dim=1)
        self.train_div(torch.softmax(y_hat, dim=1), torch.softmax(ens_logits, dim=1))
        self.train_acc(amax_hat, amax_ens)
        self.train_f1(amax_hat, amax_ens)
        self.log("train_kl_div", self.train_div, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True)
        self.log("train_logits_loss", logits_loss, on_step=False, on_epoch=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True)
        return total_loss

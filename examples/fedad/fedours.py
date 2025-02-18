import torch
from torch import nn
import torchmetrics
import lightning as pl
from argparse import Namespace
from datetime import datetime

from typing import Sequence

import utils
from gradcam import GradCAM
from server_node import (
    lam_cnn,  # noqa: F401
    lam_resnet,
    training_phase,  # noqa: F401
    load_models,
    ServerNode,
)

import sys

sys.path.append("../..")

from fluff.utils import timer
from fluff.datasets import CIFAR100Dataset
from fluff.datasets.partitions import BalancedFraction
from fluff.aggregator import FedAvg
from fluff.models import LitModel


def generate_model_run_name() -> str:
    return f"Fedours_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"


class FedAdServer(LitModel):
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
        # self.ensemble_cams = [GradCAM(mod, "layer3.2.conv2") for mod in self.ensemble]
        # self.server_cam = GradCAM(self.cnn, "layer3.2.conv2")

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

        with torch.no_grad():
            batch_logits = [ens.forward(x) for ens in self.ensemble]

        ens_logits = utils.alternative_avg(
            raw_logits=batch_logits,
            raw_statistics=self._count_stats,
            num_classes=10,
            num_nodes=len(self.ensemble),
        )

        logits_loss = self.dist_criterion(y_hat, ens_logits, T=3)

        """
        # cam_generation_start = time.time()
        class_cams, server_cams = self.cam_generation(
            batch_logits=batch_logits,
            server_logits=y_hat,
            num_samples=1,
            top=2,
        )

        # print(f"#> cam_generation {(time.time() - cam_generation_start):.4f}s")

        union_loss = utils.loss_union(
            class_cams.amax(dim=(1, 2)),
            server_cams.amax(dim=(1,)),
            num_classes=10,
        )

        inter_loss = utils.loss_intersection(
            class_cams.amin(dim=(1, 2)),
            server_cams.amin(dim=(1,)),
            num_classes=10,
        )

        total_loss = logits_loss + union_loss + inter_loss

        self.log("train_union_loss", union_loss, on_step=True, on_epoch=True)
        self.log("train_inter_loss", inter_loss, on_step=True, on_epoch=True)
        self.log("train_total_loss", total_loss, on_step=True, on_epoch=True)
        """

        self.train_div(torch.softmax(y_hat, dim=1), torch.softmax(ens_logits, dim=1))
        self.train_acc(y_hat.argmax(dim=1), ens_logits.argmax(dim=1))
        self.log("train_kl_div", self.train_div, on_step=False, on_epoch=True)
        self.log("train_acc", self.train_acc, on_step=True, on_epoch=True)
        self.log("train_logits_loss", logits_loss, on_step=False, on_epoch=True)

        # return total_loss
        return logits_loss

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


@timer
def run(args: Namespace):
    pl.seed_everything(args.seed, workers=True)
    exp_name = generate_model_run_name()

    # Training

    print(args)
    # ens, stats = training_phase(args, name="five-resnet-alpha-0_01", save=True)
    ens, stats = load_models("./models/five-resnet-alpha-0_5", lam_resnet)

    s_model = lam_resnet()

    for en in ens:
        en.freeze_bn = True  # type: ignore
        en.freeze_bn_affine = True  # type: ignore
        en.train(False)

    # Average

    aggregator = FedAvg()
    s_model.load_state_dict(aggregator.run(ens))

    # Distillation

    server = ServerNode(
        "server",
        exp_name,
        FedAdServer(
            s_model,
            ensemble=ens,
            distillation=args.distill,
        ),
        CIFAR100Dataset(
            batch_size=args.batch,
            partition=BalancedFraction(percent=0.8),
            seed=args.seed,
        ),
        num_workers=args.workers,
        seed=args.seed,
        hp=args,
    ).setup()

    print("🧫 Starting distillation")
    server.get_model().set_count_statistics(stats)

    server.train(
        args.rounds,
        dev_runs=args.dev_batches,
        skip_val=False,
        skip_test=False,
    )

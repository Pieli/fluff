import torch
from torch.utils import data
import lightning as pl
from argparse import Namespace
from datetime import datetime
from lightning.pytorch.callbacks import Timer, LambdaCallback

from server_node import fact

from models import ServerLitCifar100LogitsOnly

from base_trainer import load_models

import sys

sys.path.append("../..")

from fluff import Node
from fluff.utils import (
    timer,
    generate_confusion_matrix,
    UtilizationMonitoring,
    EarlyStoppingAtTarget,
)
from fluff.datasets import CIFAR100Dataset, CIFAR10Dataset, MNISTDataset, FMNISTDataset
from fluff.datasets.partitions import BalancedFraction
from fluff.aggregator import FedAvg
from fluff.datasets.partitions import DirichletMap


def generate_model_run_name() -> str:
    return f"Fedours_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"


@timer
def run(args: Namespace):
    pl.seed_everything(args.seed, workers=True)
    exp_name = generate_model_run_name()

    # Training

    data_cls, lam_model = fact(args.data)

    print(args)
    ens, stats = load_models(args.base, lam_model)

    s_model = lam_model()

    for en in ens:
        en.freeze_bn = True  # type: ignore
        en.freeze_bn_affine = True  # type: ignore
        en.train(False)

    # Average

    aggregator = FedAvg()
    s_model.load_state_dict(aggregator.run(ens))

    # Distillation

    def adjust_bias(model, stats):
        print(stats)
        print(model.classifier.bias)
        with torch.no_grad():
            bias_adjust = 3.5
            for ind, stat in enumerate(stats):
                if stat == 0:
                    model.classifier.bias[ind] += bias_adjust

    import copy

    copy_s_model = copy.deepcopy(s_model)

    num_nodes = range(args.nodes) if not args.select else [args.select]

    for node in num_nodes:

        dataset = data_cls(
            batch_size=args.batch,
            partition=DirichletMap(
                partition_id=node,
                partitions_number=args.nodes,
                alpha=args.alpha,
            ),
            seed=args.seed,
        )

        timer_call = (
            Timer(duration=dict(hours=args.max_time), interval="step")
            if args.conv
            else LambdaCallback()
        )
        device_stats_callback = UtilizationMonitoring()
        early_stop = (
            EarlyStoppingAtTarget(target_metric="val_f1_epoch", target_value=args.conv)
            if args.conv
            else LambdaCallback()
        )

        print(dataset.count_train())
        cs_model = copy.deepcopy(copy_s_model)
        adjust_bias(cs_model, dataset.count_train())
        print(cs_model.classifier.bias)

        print(f"[+>] Evaluating node number: {node}")
        server = Node(
            f"server-{node}",
            exp_name,
            ServerLitCifar100LogitsOnly(
                cs_model,
                ensemble=ens,
                distillation=args.distill,
                lr=args.lr,
            ),
            dataset,
            num_workers=args.workers,
            seed=args.seed,
            hp=args,
        ).setup()

        print("🧫 Starting distillation")
        server.get_model().set_count_statistics(stats)

        server.train(
            args.rounds,
            dev_runs=args.dev_batches,
            swap_val_to_test=args.use_test_loader,
            skip_val=False,
            skip_test=False,
            callbacks=[timer_call, device_stats_callback, early_stop],
        )

        if isinstance(early_stop, EarlyStoppingAtTarget):
            if early_stop.stopped:
                print(f"[+] target reached of {args.conv}, stopped early")
                return

        if args.conv and timer_call.time_remaining() <= 0.0:
            print("[+] time limit reached...")
            return

        # Confusion matrix
        test_loader = data.DataLoader(
            dataset.test_set,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
        )

        generate_confusion_matrix(
            server.get_model().cnn, test_loader, f"fedours-{args.data}-{args.alpha}"
        )

import copy
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Timer, LambdaCallback
from argparse import Namespace
from datetime import datetime

import os
import sys

sys.path.append("../..")

from fluff import Node
from fluff.utils import timer, UtilizationMonitoring
from fluff.aggregator import FedAvg
from fluff.datasets.partitions import DirichletMap


from typing import Dict, Any
from collections.abc import Mapping


from models import LitCNN, MoonModel
from server_node import fact


def generate_model_run_name() -> str:
    return f"MOON_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}"


class MyStrat(pl.pytorch.strategies.SingleDeviceStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_model_state_dict(
        self, checkpoint: Mapping[str, Any], strict: bool = True
    ) -> None:
        pass

    """
    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
         print("[!] Skipping optimizer state dict load")
        # skip entirely to avoid mismatches
    """


@timer
def run(args: Namespace):

    pl.seed_everything(args.seed, workers=True)
    exp_name = generate_model_run_name()

    agg = FedAvg()
    data_cls, model_type = fact(args.data)

    print(args)

    nodes = [
        Node(
            f"node-{num}",
            exp_name,
            MoonModel(
                model_type(),
                num_classes=10,
                lr=0.01,
                mu=5,
                tau=0.5,
            ),
            data_cls(
                batch_size=args.batch,
                partition=DirichletMap(
                    partition_id=num,
                    partitions_number=args.nodes,
                    alpha=args.alpha,
                ),
                seed=args.seed,
            ),
            seed=args.seed,
            num_workers=args.workers,
            hp=args,
        ).setup()
        for num in range(args.nodes)
    ]

    # For the evaluation of the global model
    # even though a partition is loaded it is not needed since the test set
    # stays unchanged, could change DirichletMap to NullMap
    server = Node(
        "server",
        exp_name,
        LitCNN(
            model_type(),
            num_classes=10,
            lr=1e-3,
        ),
        data_cls(
            batch_size=args.batch,
            partition=DirichletMap(
                partition_id=0,
                partitions_number=args.nodes,
                alpha=args.alpha,
            ),
        ),
        num_workers=args.workers,
        seed=args.seed,
        hp=args,
    ).setup()

    timer_call = (
        Timer(duration=dict(minutes=args.max_time), interval="step")
        if args.conv
        else LambdaCallback()
    )

    callback = [ModelCheckpoint(save_last=True) for _ in range(args.nodes)]
    device_stats_callback = [UtilizationMonitoring() for _ in range(args.nodes)]

    chk_point = f"./fluff_logs/{exp_name}"
    paths = [
        os.path.join(
            chk_point,
            node.get_name(),
            "MoonModel",
            "checkpoints",
            "last.ckpt",
        )
        for node in nodes
    ]

    new_state: Dict[str, Any] = copy.deepcopy(model_type().state_dict())
    for round in range(args.rounds):
        print(f"[+] Started round number {round + 1}")
        for ind, node in enumerate(nodes):
            print(f"[+] Started training node {node.get_name()} - Round {round + 1}")

            node._model.global_model.load_state_dict(copy.deepcopy(new_state))

            node.train(
                epochs=args.epochs * (round + 1),
                dev_runs=args.dev_batches,
                skip_val=False,
                skip_test=False,
                callbacks=[timer_call, callback[ind], device_stats_callback[ind]],
                ckpt_path=(paths[ind] if round > 0 else None),
                strat=MyStrat(device="cuda:0"),
                enable_progress_bar=True,
            )

            if args.conv and timer_call.time_remaining() <= 0.0:
                print("[+] time limit reached...")
                return

        new_state = agg.run([node.get_model().cnn for node in nodes])
        server._model.cnn.load_state_dict(copy.deepcopy(new_state))
        metrics = server.test(epochs=(args.epochs * (round + 1)))

        if args.conv and metrics[0]["test_f1"] > args.conv:
            print(f"target accuracy of {args.conv} reached")
            return

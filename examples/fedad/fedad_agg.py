import gc
import logging

import torch
import torch.nn as nn

from nebula.core.aggregation.aggregator import Aggregator
from nebula.core.datasets.nebuladataset import NebulaDataset

from nebula.core.datasets.cifar100.cifar100 import CIFAR100Dataset
from nebula.core.training.lightning import Lightning


class FedADAgg(Aggregator):
    """
    Aggregator: FedAD
    Authors: Gong et al.
    Year: 2021
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)

    def run_aggregation(self, models):

        import pickle
        with open("/tmp/models.pkl", "wb") as f:
            pickle.dump(models, f)

        models_list = list(models.values())

        dataset = CIFAR100Dataset(
            num_classes=100,
            partition_id=0,
            batch_size=32,
            partition="balancediid",
            partitions_number=1,
            partition_parameter=1,
            iid=True,
            seed=42,
            config=self.config,
        )

        fined_tuned_models = []
        for indx, model in enumerate(models_list):
            logging.info(f"Fine-tuning model: [{indx}]... ")
            fined_tuned_models.append(fine_tune(model, dataset, self.config, logging),)

            # TODO start with easiest w
            # TODO extraction of logits from models

        logging.info("Finished win!... ")

        return standard_fedavg(models)


def fine_tune(model: torch.nn.Module, data: NebulaDataset, config, logger):
    model.fc2 = torch.nn.Linear(model.fc1.out_features, 100)

    for param in model.parameters():
        param.requires_grad = False

    model.fc2.weight.requires_grad = True
    model.fc2.bias.requires_grad = True

    trainer = Lightning(model=model, data=data, config=config)
    trainer.set_epochs(10)
    trainer.create_trainer()
    trainer.train()
    logging.info("Print here")

    return model


def standard_fedavg(models):
    logging.info("Running Standard FedAvg")
    models = list(models.values())

    total_samples = float(sum(weight for _, weight in models))

    if total_samples == 0:
        raise ValueError("Total number of samples must be greater than zero.")

    last_model_params = models[-1][0]
    accum = {layer: torch.zeros_like(param, dtype=torch.float32) for layer, param in last_model_params.items()}

    with torch.no_grad():
        for model_parameters, weight in models:
            normalized_weight = weight / total_samples
            for layer in accum:
                accum[layer].add_(
                    model_parameters[layer].to(accum[layer].dtype),
                    alpha=normalized_weight,
                )

    del models
    gc.collect()

    # self.print_model_size(accum)
    return accum

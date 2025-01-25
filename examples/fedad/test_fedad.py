import utils
from fluff.partitions import NullMap
from fluff import Node
from datasets import CIFAR100Dataset
from models import ServerLitCNNCifar100, LitCNN, CNN

import sys
sys.path.append('../..')


def test_logits_are_extracted_correctly():
    nodes = [Node(f"test-{num}",
             LitCNN().set_distillation(True),
             CIFAR100Dataset(batch_size=500, partition=NullMap(partition_id=0, partitions_number=1)), seed=420).setup()
             for num in range(2)]

    server = Node("server", ServerLitCNNCifar100(CNN(), distillation_phase=False),
                  CIFAR100Dataset(
                      batch_size=500,
                      partition=NullMap(
                          partition_id=0,
                          partitions_number=1
                      )),
                  seed=420).setup()

    print("\N{Flexed Biceps} Pre-Training server")
    # server.train(args.epochs, args.dev_batches)

    print("🧫 Starting distillation")
    server.get_model().set_distillation_phase(True)

    for round in range(1):
        round_logits = []
        round_counts = []
        for node in nodes:
            print(f"Training {node.get_name()}")
            node.train(1, None)
            round_logits.append(node.get_model().get_average_logits())
            round_counts.append(node.get_model().get_class_counts())

        batch_logits = list(zip(*round_logits))
        batch_counts = list(zip(*round_counts))

        assert len(batch_logits) > 0
        assert len(batch_counts) > 0

        ens_logits = [utils.logits_ensemble_eq_3(log, count, 100, 2)
                      for log, count in zip(batch_logits, batch_counts)]

        assert len(ens_logits) > 0

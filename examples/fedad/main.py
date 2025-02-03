from argparse import ArgumentParser

import runner
import fedavg

parser = ArgumentParser()

parser.add_argument("-e", "--epochs", type=int, default=3)
parser.add_argument("-k", "--nodes", type=int, default=1)
parser.add_argument("-b", "--batch", type=int, default=128)
parser.add_argument("-t", "--rounds", type=int, default=10)


parser.add_argument("-w", "--workers", type=int, default=4)

parser.add_argument("--dev-batches", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)


args = parser.parse_args()

# runner.run(args)
fedavg.run(args)

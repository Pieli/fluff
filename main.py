from argparse import ArgumentParser

import runner

parser = ArgumentParser()

parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--nodes", type=int, default=1)
parser.add_argument("--local_batch", type=int, default=128)
parser.add_argument("--rounds", type=int, default=128)


if __name__ == "__main__":
    args = parser.parse_args()

    runner.run(args)

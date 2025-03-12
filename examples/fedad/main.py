from argparse import ArgumentParser

import fedad
import fedavg
import fedours
import tinyad
import tinyours

import lmd
import expours
import combined

import base_trainer

parser = ArgumentParser()


parser.add_argument("mode", type=str)

parser.add_argument("-e", "--epochs", type=int, default=3)
parser.add_argument("-k", "--nodes", type=int, default=1)
parser.add_argument("-b", "--batch", type=int, default=128)
parser.add_argument("-t", "--rounds", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-3)

parser.add_argument("--alpha", type=float, default=0.5)
parser.add_argument("--distill", type=str, default="kl")

parser.add_argument("-w", "--workers", type=int, default=4)

parser.add_argument("--dev-batches", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)

parser.add_argument("--base", type=str, default="five-resnet-alpha-0_5")

args = parser.parse_args()

match (args.mode):
    case "fedavg":
        fedavg.run(args)
    case "fedad":
        fedad.run(args)
    case "fedours":
        fedours.run(args)
    case "expours":
        expours.run(args)
    case "combined":
        combined.run(args)
    case "lmd":
        lmd.run(args)
    case "tinyad":
        tinyad.run(args)
    case "tinyours":
        tinyours.run(args)
    case "train":
        base_trainer.training_phase(args, args.base, save=True)
    case _:
        print("Arguement [mod] was not valid")

import urllib.parse
import requests
import statistics
from multiprocessing import Pool
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("filename")
parser.add_argument("-m", "--method", type=str, default="f1")
parser.add_argument("-b", "--backend", type=str, default="server")

args = parser.parse_args()
input_dir = args.filename

p = Path(input_dir)
log_dirs: list[Path] = [l_dir for l_dir in p.iterdir() if l_dir.is_dir()]


def scalar_values(logdir: str, scalar_name: str):
    print(f"processing: {logdir}")
    ea = event_accumulator.EventAccumulator(
        logdir,
        size_guidance={
            "scalars": 0,
        },
    )
    ea.Reload()

    events = ea.Scalars(scalar_name)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    return steps, values


def visualize_f1_over_epoch(values):
    sns.set(style="darkgrid")
    plt.figure(figsize=(8, 5))

    for dir, res in sorted(values, key=lambda x: x[0]):
        sns.lineplot(x=res[0], y=res[1], label=dir)

    plt.xlabel("Step")
    plt.ylabel("F1-score")
    ax = plt.gca()
    ax.set_ylim([0, 1.1])
    ax.axhline(y=1.0, linewidth=1, color="k", ls="--")

    # plt.show()
    plt.savefig(f"{p.name}-{scalar}.png")


def print_single_f1(values):
    val_dict = dict(values)
    keys = sorted(val_dict.keys())
    print(keys)

    vals = [val_dict[key][1][0] for key in keys]
    print(vals)

    print(" & ".join(str(round(val, 5)) for val in vals))
    print(f"Max: {round(max(vals), 5)}")
    mean = statistics.mean(vals) * 100
    stdv = statistics.stdev(vals) * 100
    print(f"${mean:0.2f}\\pm{stdv:0.2f}$")


def request_scalar(logdir: str, scalar: str):
    print(f"processing: {logdir}")
    run = logdir
    if logdir.startswith("fluff_logs/"):
        run = run.partition("fluff_logs/")[2]

    url_run = urllib.parse.quote_plus(run)

    tensorboard_url = "http://localhost:5000"
    try:
        response = requests.get(
            f"{tensorboard_url}/data/plugin/scalars/scalars?tag={scalar}&run={url_run}"
        )
    except requests.exceptions.ConnectionError:
        print(f"Failed to connect to {tensorboard_url}")
        print("Is the tensorboard server running?")
        raise RuntimeError

    # Returns a list of [wall_time, step, value]
    data = response.json()
    sorted_data = sorted(data, key=lambda x: x[1])
    res = list(zip(*sorted_data))
    return (res[1], res[2])


if __name__ == "__main__":
    dirs = []
    for l_dir in log_dirs:
        children = list(l_dir.iterdir())
        if len(children) != 1:
            print("Each node should only have one child")
            exit(1)

        dir = children[0]
        if not dir.exists():
            print(f"Path: {dir} does not exist")
            exit(1)
        dirs.append(dir)

    match args.method:
        case "f1":
            scalar = "test_f1"
            func = print_single_f1
        case "f1oe":
            scalar = "val_f1_epoch"
            # func = visualize_f1_over_epoch
            func = print
        case _:
            print(f"No action found for {args.method}")
            exit(1)

    def local_f1(logdir: Path):
        return (str(logdir.parent.name), scalar_values(str(logdir), scalar))

    def server_f1(logdir: Path):
        return (str(logdir.parent.name), request_scalar(str(logdir), scalar))

    match args.backend:
        case "local":
            backend_func = local_f1
        case "server":
            backend_func = server_f1
        case _:
            print(f"No action found for {args.backend}")
            exit(1)

    try:
        with Pool(processes=5) as pool:
            results = pool.map(backend_func, dirs)
    except RuntimeError:
        print("One of the processes terminated")
        exit(0)

    print("*" * 40)
    print()

    func(results)

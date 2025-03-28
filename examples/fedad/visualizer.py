import sys
from multiprocessing import Pool
from pathlib import Path, PosixPath
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt
import seaborn as sns


args = sys.argv
if len(args) < 2:
    print(f"{__file__} [logdir]")
    exit(1)


input_dir = args[1]

p = Path(input_dir)
log_dirs: list[PosixPath] = [l_dir for l_dir in p .iterdir() if l_dir.is_dir()]


def scalar_values(logdir: str, scalar_name: str):
    print(f"processing: {logdir}")
    ea = event_accumulator.EventAccumulator(
        logdir, size_guidance={
            "scalars": 0,
        })
    ea.Reload()

    scalar_name = "val_f1_epoch"
    events = ea.Scalars(scalar_name)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    return steps, values


sns.set(style="darkgrid")
plt.figure(figsize=(8, 5))

dirs = []
for l_dir in log_dirs:
    dir = l_dir / "LitCNN"
    if not dir.exists():
        print(f"Path: {dir} does not exist")
    dirs.append(dir)

scalar = "val_f1_epoch"


def get_f1(logdir: PosixPath):
    return scalar_values(str(logdir), scalar)


with Pool(processes=5) as pool:
    results = pool.map(get_f1, dirs)


for dir, res in zip(dirs, results):
    sns.lineplot(x=res[0], y=res[1], label=f"{dir.parent.name}")

plt.xlabel("Step")
plt.ylabel("Value")
plt.legend()

plt.savefig(f"{p.name}-{scalar}.png")
# plt.show()

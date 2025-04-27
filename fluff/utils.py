import os
import functools
import time
import torch

import seaborn as sns
import matplotlib.pyplot as plt

import lightning as pl

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


import pynvml
import psutil


class EarlyStoppingAtTarget(pl.Callback):
    def __init__(self, target_metric: str, target_value: float, mode: str = "max"):
        self.target_metric = target_metric
        self.target_value = target_value
        self.mode = mode
        self.stopped = False

    def on_validation_epoch_end(self, trainer, pl_module):
        """
        Checks if the target metric has reached the target value at the end of each validation.
        """
        if self.stopped:
            return  # Training already stopped, no need to check further

        # Get the current value of the target metric
        cur_metric_val = trainer.callback_metrics.get(self.target_metric, None)

        if cur_metric_val is not None:
            if self.mode == "max" and cur_metric_val >= self.target_value:
                trainer.should_stop = True
                self.stopped = True


class UtilizationMonitoring(pl.Callback):
    def __init__(self, gpu_index=0):
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)

        gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        gpu_mem_percent = round(gpu_mem.used / gpu_mem.total * 100, 3)

        kwargs = {"prog_bar": False, "on_step": True, "on_epoch": False}

        # GPU usage
        power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.00
        pl_module.log("resources/gpu_utilization", util.gpu, **kwargs)
        pl_module.log("resources/gpu_memory_used", gpu_mem.used, **kwargs)
        pl_module.log("resources/gpu_memory_percentage", gpu_mem_percent, **kwargs)
        pl_module.log("resources/gpu_power", power, **kwargs)

        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        pl_module.log("resources/cpu_percent", cpu_percent, **kwargs)

        # RAM Usage
        ram = psutil.virtual_memory()
        ram_percent = ram.percent  # In percentage
        pl_module.log("resources/ram_percent", ram_percent, **kwargs)


def generate_confusion_matrix(model, dataloader, name: str, device="cuda"):
    model.eval()
    model.to(device)

    os.makedirs("results", exist_ok=True)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=None)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"results/{name}.png", bbox_inches="tight", pad_inches=0.05, dpi=100)


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f"Finished {func.__name__}() in {run_time:.4f} secs")
        return value

    return wrapper_timer


def print_args(func):
    """This decorator dumps out the arguments passed to a function before calling it"""
    argnames = func.__code__.co_varnames

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        kwargs_str = ", " + kwargs.items() if kwargs else ""
        print(
            func.__name__,
            "(",
            ", ".join(f"{name}={arg}" for name, arg in zip(argnames, args)),
            kwargs_str,
            ")",
            sep="",
        )
        return func(*args, **kwargs)

    return wrapper


def plot_tuning(inputs: list[dict], x_label: str, show=True, save=False):
    """
    Example:
        plot_tuning([{label: ..., lr: ..., loss: ... }, {...}], x_label = "lr")
    """
    assert isinstance(inputs, list)
    assert len(inputs) > 0

    sns.set(style="darkgrid")

    plt.figure(figsize=(8, 5))

    labels = list(inputs[0].keys())
    labels.remove(x_label)
    labels.remove("label")

    if len(labels) > 1:
        raise ValueError(f"Too many keys provided: found {len(labels)}")

    y_label = labels[0]

    for elem in inputs:
        if x_label not in elem:
            raise ValueError(f"x_label: {x_label} not in {elem}")

        if y_label not in elem:
            raise ValueError(f"y_label: {y_label} not in {elem}")

        sns.lineplot(x=elem[x_label], y=elem[y_label], label=elem["label"])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()

    if show:
        plt.show()

    if save:
        # file = "lr_tune_" + str(round(time.time())) + ".png"
        file = "lr_tune.png"
        print(f"[*] Saving to file... {file}")
        plt.savefig(file)

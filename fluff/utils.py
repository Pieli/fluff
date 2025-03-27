import functools
import time

import seaborn as sns
import matplotlib.pyplot as plt


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

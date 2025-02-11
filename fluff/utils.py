import functools
import time


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

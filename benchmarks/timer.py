"""
benchmarks/timer.py

Utilities for measuring execution time.

This module provides:
  - timed: a decorator that prints how long a function takes to run
  - benchmark: a function that runs another function multiple times and returns
    the mean elapsed time in milliseconds
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable


def timed(fn: Callable) -> Callable:
    """
    Decorator that measures and prints the execution time of a function
    in milliseconds.
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        print(f"{fn.__name__} took {elapsed_ms:.3f} ms")

        return result

    return wrapper


def benchmark(
    fn: Callable,
    *args: Any,
    repeats: int = 5,
    **kwargs: Any,
) -> float:
    """
    Run a function multiple times and return the mean elapsed time
    in milliseconds.
    """
    if repeats <= 0:
        raise ValueError("repeats must be greater than 0")

    elapsed_times = []

    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        elapsed_times.append(elapsed_ms)

    return sum(elapsed_times) / len(elapsed_times)
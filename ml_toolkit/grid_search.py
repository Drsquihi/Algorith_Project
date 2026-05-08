"""
ml_toolkit/grid_search.py

Exhaustive hyperparameter grid search backed by a HyperparamRegistry.

This module tries every possible hyperparameter combination from a parameter
grid, evaluates each configuration, and stores the result inside a BST-backed
HyperparamRegistry.

Algorithmic concepts used:
  - itertools.product  : brute-force exhaustive search over all combinations
  - zip                : rebuilds each tuple combination into a params dict
  - HyperparamRegistry : stores trial scores in a BST
  - tqdm               : optional progress bar
"""

from __future__ import annotations

import itertools
from typing import Any, Callable

from tqdm import tqdm

from bst_toolkit import HyperparamRegistry


def grid_search(
    param_grid: dict[str, list[Any]],
    evaluate_fn: Callable[[dict[str, Any], Any], float],
    dataset: Any,
    verbose: bool = True,
) -> HyperparamRegistry:
    """
    Run an exhaustive hyperparameter grid search.

    Parameters
    ----------
    param_grid:
        Dictionary mapping hyperparameter names to lists of candidate values.

        Example:
        {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
        }

    evaluate_fn:
        Function that receives one parameter combination and the dataset.

        Expected form:
            evaluate_fn(params, dataset) -> float

    dataset:
        Any object understood by evaluate_fn.
        Usually this is a tuple like (X, y).

    verbose:
        If True, show a tqdm progress bar.

    Returns
    -------
    HyperparamRegistry
        A BST-backed registry containing the evaluated trials.

    Notes
    -----
    Scores are rounded to 6 decimals before insertion to reduce floating-point
    noise. If two trials produce the same rounded score, the BST keeps the first
    inserted one, following the project collision rule.
    """
    registry = HyperparamRegistry()

    if not param_grid:
        return registry

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for name, values in param_grid.items():
        if not isinstance(values, list):
            raise TypeError(
                f"Parameter '{name}' must map to a list of values, "
                f"but got {type(values).__name__}."
            )

        if len(values) == 0:
            raise ValueError(f"Parameter '{name}' has an empty list of values.")

    combinations = list(itertools.product(*param_values))

    iterator = (
        tqdm(combinations, desc="Grid search", unit="trial")
        if verbose
        else combinations
    )

    for combo in iterator:
        params = dict(zip(param_names, combo))

        score = evaluate_fn(params, dataset)
        score = round(float(score), 6)

        registry.add_trial(score, params)

        if verbose:
            best_trial = registry.best()
            if best_trial is not None:
                iterator.set_postfix(best=f"{best_trial.score:.6f}")

    return registry
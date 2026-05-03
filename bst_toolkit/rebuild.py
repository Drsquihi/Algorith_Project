from __future__ import annotations
from typing import List, Callable, Tuple
from .registry import HyperparamRegistry
from .node import TrialNode
import random


def rebuild_naive(
    registry: HyperparamRegistry,
    evaluate_fn: Callable,
    new_dataset
) -> HyperparamRegistry:
    """
    Strategy 1 — Re-score every trial and insert one by one.
    """
    new_registry = HyperparamRegistry()

    for node in registry.all_trials():
        new_score = evaluate_fn(node.params, new_dataset)
        new_registry.add_trial(new_score, node.params)

    return new_registry


def rebuild_shuffled(
    registry: HyperparamRegistry,
    evaluate_fn: Callable,
    new_dataset
) -> HyperparamRegistry:
    """
    Strategy 2 — Shuffle trials before re-inserting.
    """
    new_registry = HyperparamRegistry()

    trials = registry.all_trials()
    random.shuffle(trials)

    for node in trials:
        new_score = evaluate_fn(node.params, new_dataset)
        new_registry.add_trial(new_score, node.params)

    return new_registry


def rebuild_balanced(
    registry: HyperparamRegistry,
    evaluate_fn: Callable,
    new_dataset
) -> HyperparamRegistry:
    """
    Strategy 3 — Build a perfectly balanced BST using divide & conquer.
    """
    scored_trials = []

    for node in registry.all_trials():
        new_score = evaluate_fn(node.params, new_dataset)
        scored_trials.append((round(new_score, 6), node.params))

    scored_trials.sort(key=lambda item: item[0])

    new_registry = HyperparamRegistry()
    new_registry._bst.root = _build_from_sorted(scored_trials)
    new_registry._bst._size = len(scored_trials)
    new_registry._history = scored_trials

    return new_registry


def _build_from_sorted(sorted_trials: List[Tuple[float, dict]]):
    """
    Recursively build a balanced BST from a sorted list of (score, params).
    """
    if not sorted_trials:
        return None

    mid = len(sorted_trials) // 2
    score, params = sorted_trials[mid]

    root = TrialNode(score, params)
    root.left = _build_from_sorted(sorted_trials[:mid])
    root.right = _build_from_sorted(sorted_trials[mid + 1:])

    return root
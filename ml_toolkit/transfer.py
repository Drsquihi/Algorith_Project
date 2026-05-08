"""
ml_toolkit/transfer.py

Compare hyperparameter rankings between two HyperparamRegistry objects.

Usually:
  - registry_A contains trials from Dataset A: Breast Cancer Wisconsin
  - registry_B contains the same configurations re-evaluated on Dataset B: Banknote

Algorithmic concepts used:
  - In-order traversal : gets all BST nodes sorted by score
  - Hash table / dict  : builds fast rank lookups
  - Sorting            : sorts the final report by rank drift
"""

from __future__ import annotations

from typing import Any

from bst_toolkit import HyperparamRegistry


def _params_key(params: dict[str, Any]) -> tuple:
    """
    Convert a params dictionary into a hashable key.

    Example:
        {"n_estimators": 100, "max_depth": 5}

    Becomes:
        (("max_depth", 5), ("n_estimators", 100))

    This lets us compare the same configuration across two registries.
    """
    return tuple(sorted(params.items()))


def analyse_transfer(
    registry_A: HyperparamRegistry,
    registry_B: HyperparamRegistry,
) -> list[dict[str, Any]]:
    """
    Compare rankings of the same configurations between two registries.

    Returns a ranked report with:
      - params
      - score_A
      - score_B
      - rank_A
      - rank_B
      - drift
      - transfer

    drift = rank_A - rank_B

    Positive drift means the configuration improved on Dataset B.
    Negative drift means the configuration degraded on Dataset B.
    Zero drift means the configuration stayed stable.
    """
    trials_A = registry_A.all_trials()
    trials_B = registry_B.all_trials()

    ranked_A = list(reversed(trials_A))
    ranked_B = list(reversed(trials_B))

    ranks_A = {}
    for rank, node in enumerate(ranked_A, start=1):
        key = _params_key(node.params)
        ranks_A[key] = {
            "params": node.params,
            "score_A": node.score,
            "rank_A": rank,
        }

    ranks_B = {}
    for rank, node in enumerate(ranked_B, start=1):
        key = _params_key(node.params)
        ranks_B[key] = {
            "score_B": node.score,
            "rank_B": rank,
        }

    report = []

    for key, info_A in ranks_A.items():
        if key not in ranks_B:
            continue

        info_B = ranks_B[key]

        rank_A = info_A["rank_A"]
        rank_B = info_B["rank_B"]
        drift = rank_A - rank_B

        if drift > 0:
            transfer_label = "good"
        elif drift == 0:
            transfer_label = "stable"
        else:
            transfer_label = "poor"

        report.append(
            {
                "params": info_A["params"],
                "score_A": info_A["score_A"],
                "score_B": info_B["score_B"],
                "rank_A": rank_A,
                "rank_B": rank_B,
                "drift": drift,
                "transfer": transfer_label,
            }
        )

    return sorted(report, key=lambda row: row["drift"], reverse=True)
# BST-Backed Hyperparameter Optimiser with Transfer Analysis

> **Course:** Algorithmic Workshop — AIS / EPITA  
> **Instructor:** Adrian ROSARI  
> **Format:** Python packages + Jupyter Notebook  

---

## Team Members

| Name | Role |
|---|---|
| **Amjad Bsat** | Developer |
| **Majd Hamoud** | Developer |

---

## Table of Contents

1. [Project Purpose](#1-project-purpose)
2. [Core Concepts](#2-core-concepts)
3. [Project Architecture](#3-project-architecture)
4. [File Structure](#4-file-structure)
5. [Datasets](#5-datasets)
6. [Package Reference](#6-package-reference)
   - [bst_toolkit](#61-bst_toolkit--the-algorithmic-core)
   - [ml_toolkit](#62-ml_toolkit--machine-learning-layer)
   - [benchmarks](#63-benchmarks--timing-utilities)
7. [Installation](#7-installation)
8. [Usage Guide](#8-usage-guide)
9. [How the Two Phases Connect](#9-how-the-two-phases-connect)
10. [Algorithm Complexity Summary](#10-algorithm-complexity-summary)
11. [Grading Alignment](#11-grading-alignment)

---

## 1. Project Purpose

This project is a **complete, installable data science tool** built from scratch. It is not a collection of isolated exercises — it is a real software system that integrates classical algorithm design (Binary Search Trees, recursion, divide & conquer, hashing, sorting) with practical machine learning (hyperparameter tuning, cross-validation, transfer analysis).

### The Problem It Solves

When training machine learning models, practitioners often run a **grid search**: they define a set of candidate hyperparameters (e.g. number of trees, maximum depth), evaluate every combination, and pick the best one. The standard approach stores results in a flat list or a pandas DataFrame — which is perfectly fine for retrieval, but loses all structural information about how configurations *relate* to each other by score.

This project replaces that flat structure with a **Binary Search Tree (BST) keyed by accuracy score**. This choice is deliberate:

- **Retrieval by score range** becomes O(log n + k) instead of O(n).
- **Top-k configurations** can be extracted in O(k + h) via a reverse in-order traversal, without sorting.
- **Minimum and maximum scores** are found in O(h) by traversing to the leftmost or rightmost leaf.
- The **in-order traversal** of the BST automatically yields configurations sorted by score — a property used extensively in the transfer analysis phase.

### The Two Phases

**Phase 1 — Grid Search on Dataset A (Breast Cancer Wisconsin)**  
Every combination in the hyperparameter grid is evaluated on Dataset A using 5-fold cross-validation. Each trial's result (accuracy score + hyperparameter dict) is inserted into a `HyperparamRegistry` (a BST wrapper). After the search, the tree holds the complete ranked history of all trials.

**Phase 2 — Transfer Analysis on Dataset B (Banknote Authentication)**  
The registry from Phase 1 is reloaded. Every configuration is re-evaluated on Dataset B — a structurally very different dataset (30 features → 4 features, medical → signal processing). The tree is rebuilt under the new scores using one of three strategies (naive, shuffled, or balanced), and a transfer report is generated that shows which configurations **generalised well** between the two domains and which degraded.

This two-phase design forces the question: *do the hyperparameters that work best on a high-dimensional medical dataset also work on a compact signal-processing dataset?* The answer, quantified as rank drift per configuration, is the analytical output of the project.

---

## 2. Core Concepts

Each module in the project maps directly to a session of the Algorithmic Workshop course.

| Concept | Where It Appears | Course Session |
|---|---|---|
| Big-O analysis, brute-force search | `grid_search.py` — `itertools.product` over all combinations | Session 1 |
| Hash tables | `transfer.py` — O(1) rank lookup dict built from BST traversal | Session 1 |
| Functional programming, decorators | `timer.py` — `@timed` with `functools.wraps` | Session 1 |
| Recursion, divide & conquer | `rebuild.py` — `_build_from_sorted`, `bst.py` recursive helpers | Session 2 |
| Sorting, insertion order effects | `rebuild.py` — naive vs shuffled vs balanced strategies | Session 3 |
| Binary search, range pruning | `registry.py` — `range_query`, `prune_below` | Session 4 |
| BST traversals (all four orders) | `bst.py` — inorder, preorder, postorder, level-order | Session 5 |

---

## 3. Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        capstone.ipynb                           │
│                   (orchestration & narrative)                   │
└───────────┬────────────────────┬────────────────────────────────┘
            │                    │
            ▼                    ▼
  ┌──────────────────┐  ┌──────────────────────┐
  │   ml_toolkit     │  │     benchmarks        │
  │  grid_search.py  │  │      timer.py         │
  │  transfer.py     │  │  @timed, benchmark()  │
  └────────┬─────────┘  └──────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────────┐
  │              bst_toolkit                 │
  │                                          │
  │  HyperparamRegistry  ──►  BST            │
  │  (registry.py)            (bst.py)       │
  │                               │          │
  │                           TrialNode      │
  │                           (node.py)      │
  │                                          │
  │  rebuild.py  (3 rebuild strategies)      │
  └──────────────────────────────────────────┘
           │
           ▼
  ┌──────────────────┐
  │      data/       │
  │  download.py     │
  │  wdbc.csv        │
  │  banknote.csv    │
  └──────────────────┘
```

The dependency graph flows strictly downward: the notebook depends on all three packages; `ml_toolkit` depends on `bst_toolkit`; `bst_toolkit` has no internal dependencies. This makes each layer independently testable.

---

## 4. File Structure

```
capstone_project/
│
├── README.md                        ← You are here
├── setup.py                         ← Editable install config (pip install -e .)
├── requirements.txt                 ← All Python dependencies with version pins
├── .gitignore                       ← Excludes __pycache__, .venv, *.pkl, etc.
│
├── data/
│   ├── download.py                  ← Idempotent download + cleaning script
│   ├── wdbc.csv                     ← Generated by download.py (Dataset A)
│   └── banknote.csv                 ← Generated by download.py (Dataset B)
│
├── bst_toolkit/                     ← Package 1: algorithmic core
│   ├── __init__.py                  ← Exports TrialNode, BST, HyperparamRegistry, rebuild
│   ├── node.py                      ← TrialNode dataclass (score, params, left, right)
│   ├── bst.py                       ← BST class: insert, delete, search, 4 traversals
│   ├── registry.py                  ← HyperparamRegistry: top-k, range query, prune, summary
│   └── rebuild.py                   ← 3 rebuild strategies + _build_from_sorted
│
├── ml_toolkit/                      ← Package 2: machine learning layer
│   ├── __init__.py                  ← Exports grid_search, analyse_transfer
│   ├── grid_search.py               ← Exhaustive grid search → HyperparamRegistry
│   └── transfer.py                  ← Cross-dataset rank drift analysis
│
├── benchmarks/                      ← Package 3: timing utilities
│   ├── __init__.py                  ← Exports timed, benchmark
│   └── timer.py                     ← @timed decorator + benchmark() function
│
└── notebook/
    └── capstone.ipynb               ← Main deliverable: full project narrative
```

### Key Design Decisions

**`setup.py` + `pip install -e .`**  
All three packages are installed in editable mode, meaning Python resolves `from bst_toolkit import ...` to the actual source files. Any edit to the source is reflected immediately without reinstalling. This mirrors professional package development practice.

**`data/download.py` is idempotent**  
Re-running the script does nothing if the CSVs already exist. This makes the project reproducible on any machine with a single command: `python data/download.py`.

**`*.pkl` excluded from Git**  
Pickled registry snapshots (if used for caching) are excluded via `.gitignore` because they are regenerable from the source data and their binary format produces unreadable diffs.

---

## 5. Datasets

Both datasets are sourced from the **UCI Machine Learning Repository** — a standard academic benchmark collection. They were chosen specifically because they are both binary classification tasks, yet structurally very different, making the transfer analysis meaningful.

### Dataset A — Breast Cancer Wisconsin (Diagnostic)

| Property | Value |
|---|---|
| **Source** | UCI ML Repository, ID 17 |
| **URL** | https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip |
| **Raw file** | `wdbc.data` (no header, comma-separated) |
| **Cleaned file** | `data/wdbc.csv` |
| **Samples** | 569 |
| **Features** | 30 real-valued measurements |
| **Target** | Column 2: `M` (malignant) → `1`, `B` (benign) → `0` |
| **Class balance** | ~37% malignant, ~63% benign |

**Feature description:**  
Each of the 30 features is one of 10 nuclear measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension) reported across 3 statistics: mean, standard error (`se`), and worst (largest) value. Column 1 (patient ID) is dropped during preprocessing.

**Preprocessing applied:**
- Drop the `id` column
- Encode `diagnosis`: `M → 1`, `B → 0`
- Standardise all 30 features with `sklearn.preprocessing.StandardScaler` (zero mean, unit variance)

### Dataset B — Banknote Authentication

| Property | Value |
|---|---|
| **Source** | UCI ML Repository, ID 267 |
| **URL** | https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip |
| **Raw file** | `data_banknote_authentication.txt` (no header, comma-separated) |
| **Cleaned file** | `data/banknote.csv` |
| **Samples** | 1,372 |
| **Features** | 4 real-valued statistics |
| **Target** | Last column: `0` (forged), `1` (genuine) |
| **Class balance** | ~56% genuine, ~44% forged |

**Feature description:**  
Features are derived from a **Discrete Wavelet Transform (DWT)** of greyscale images of banknotes: variance, skewness, curtosis, and entropy of the wavelet coefficients. The target is already binary (no encoding needed).

**Preprocessing applied:**
- Assign column names (file has no header)
- Standardise the 4 features with `StandardScaler`

### Why These Two Datasets?

The transfer analysis is designed to answer a real question: *do hyperparameters that are optimal for one dataset generalise to another?* The A → B pair creates an interesting stress test:

| Dimension | Dataset A | Dataset B |
|---|---|---|
| Feature count | 30 | 4 |
| Domain | Medical imaging (cell nuclei) | Signal processing (wavelets) |
| Samples | 569 | 1,372 |
| Feature scale | Mixed (area ~hundreds, fractal ~0.05) | Standardised wavelet stats |
| Noise profile | Biological variation | Measurement/image noise |

A configuration that does well on 30-feature medical data because of its regularisation or depth may behave completely differently on 4 highly correlated wavelet features. This structural mismatch is what the `transfer` column in the report quantifies.

---

## 6. Package Reference

### 6.1 `bst_toolkit` — The Algorithmic Core

#### `node.py` — `TrialNode`

A Python `dataclass` representing one node in the BST.

```python
@dataclass
class TrialNode:
    score:  float                    # BST key — the accuracy score
    params: dict                     # Hyperparameter configuration
    left:   Optional[TrialNode]      # Left child  (score < this node)
    right:  Optional[TrialNode]      # Right child (score > this node)
```

`__lt__` enables direct comparison (`node_a < node_b`) based on score.  
`__repr__` produces readable output: `TrialNode(score=0.9533, params={...})`.

---

#### `bst.py` — `BST`

The core data structure. All operations satisfy the BST invariant: **left child score < parent score < right child score**.

| Method | Description | Complexity |
|---|---|---|
| `insert(score, params)` | Insert a new trial. First-inserted wins on collision. | O(h) |
| `delete(score)` | Remove by score. Handles 0, 1, and 2-child cases. | O(h) |
| `search(score)` | Return matching node or `None`. | O(h) |
| `find_min()` | Leftmost node = lowest score. | O(h) |
| `find_max()` | Rightmost node = highest score. | O(h) |
| `height()` | Tree height (0 for empty). | O(n) |
| `is_balanced()` | True if `\|left_h − right_h\| ≤ 1` at every node. | O(n) |
| `inorder()` | Left → Node → Right. Returns nodes **ascending by score**. | O(n) |
| `preorder()` | Node → Left → Right. Used to serialise the tree. | O(n) |
| `postorder()` | Left → Right → Node. Used to delete the tree safely. | O(n) |
| `level_order()` | BFS level by level using `collections.deque`. | O(n) |

*h = tree height (O(log n) average, O(n) worst case for a degenerate tree).*

**Delete — the three cases:**

```
Case 1 (leaf):          Case 2 (one child):      Case 3 (two children):
    D                       D                         D
   / \          →          /       →          successor(D)
  -   -                   L                       /         \
                                              [left]    [right minus successor]
```

For Case 3, the in-order successor (minimum of the right subtree) is copied into the deleted node, then removed from the right subtree — preserving the BST invariant.

---

#### `registry.py` — `HyperparamRegistry`

A high-level API wrapping `BST`. This is the class used throughout the notebook and by `grid_search`.

| Method | Description |
|---|---|
| `add_trial(score, params)` | Insert into BST + append to history log. Rounds to 6 decimals. |
| `best()` | Highest-scoring trial (`find_max`). |
| `worst()` | Lowest-scoring trial (`find_min`). |
| `top_k(k)` | k best trials descending — reverse in-order (Right → Node → Left), stops at k. |
| `range_query(lo, hi)` | All trials with `lo ≤ score ≤ hi`, sorted ascending. Uses BST pruning. |
| `prune_below(threshold)` | Delete all trials below threshold. Returns count deleted. |
| `all_trials()` | All trials sorted ascending (in-order traversal). |
| `summary()` | Dict: count, best/worst/mean score, tree height, is_balanced. |

---

#### `rebuild.py` — Three Rebuild Strategies

After Phase 1, the registry built on Dataset A needs to be re-evaluated on Dataset B. Three strategies are offered, with deliberately different performance characteristics:

**Strategy 1 — `rebuild_naive`**

Re-scores trials in sorted (ascending) order and inserts them one by one.

```
Sorted insertion into a BST = degenerate tree (linked list):

  0.70
     \
    0.75
        \
       0.80
           \
          0.85  ← height = n, not log(n)
```

This is *intentionally broken* — it demonstrates the O(n²) worst case that motivated the other strategies. The benchmark section of the notebook measures this directly.

**Strategy 2 — `rebuild_shuffled`**

Shuffles the trial list before re-inserting (`random.shuffle`). Breaking sorted order prevents the degenerate case. Expected O(n log n) total, but balance is not guaranteed.

**Strategy 3 — `rebuild_balanced`**

Uses a divide & conquer algorithm identical in structure to merge sort:

```python
def _build_from_sorted(sorted_trials):
    mid = len(sorted_trials) // 2
    node = TrialNode(score=sorted_trials[mid][0], ...)
    node.left  = _build_from_sorted(sorted_trials[:mid])
    node.right = _build_from_sorted(sorted_trials[mid+1:])
    return node
```

Guaranteed height = ⌊log₂ n⌋. Always produces `is_balanced() == True`.  
**This is the recommended strategy for Phase 2.**

---

### 6.2 `ml_toolkit` — Machine Learning Layer

#### `grid_search.py` — `grid_search()`

```python
def grid_search(
    param_grid:  dict,       # {"n_estimators": [50, 100, 200], "max_depth": [3, 5]}
    evaluate_fn: callable,   # function(params: dict, dataset) -> float
    dataset,                 # (X, y) tuple or any object evaluate_fn understands
    verbose:     bool = True # tqdm progress bar with live best-score display
) -> HyperparamRegistry
```

Uses `itertools.product(*param_grid.values())` to generate the full cartesian product, then zips keys back onto each combination to reconstruct named dicts. This is a **brute-force exhaustive search** — every combination is evaluated with no pruning or early stopping. Scores are rounded to 6 decimal places before insertion to prevent floating-point collisions in the BST.

Example with 3 parameters of sizes [3, 4, 3]:
```
Total combinations = 3 × 4 × 3 = 36 trials
```

---

#### `transfer.py` — `analyse_transfer()`

```python
def analyse_transfer(
    registry_a:      HyperparamRegistry,
    registry_b:      HyperparamRegistry,
    drift_threshold: float = 2.0         # rank positions to classify as good/poor
) -> List[dict]
```

**Algorithm (step by step):**

```
1. registry_a.all_trials()  → in-order traversal → rank_map_a  [O(n)]
   {"n_estimators=100|max_depth=5": {"score": 0.953, "rank": 36}, ...}

2. registry_b.all_trials()  → in-order traversal → rank_map_b  [O(n)]

3. For each config in both maps:
   drift       = rank_a − rank_b          (+ = improved, - = declined)
   score_delta = score_b − score_a
   transfer    = "✓ good"  if drift >  threshold
               = "✗ poor"  if drift < −threshold
               = "~ stable" otherwise

4. sorted(report, key=lambda r: r["drift"], reverse=True)  [O(n log n)]
```

**Output columns per entry:**

| Column | Type | Meaning |
|---|---|---|
| `params` | dict | Original hyperparameter configuration |
| `score_a` | float | Accuracy on Dataset A |
| `score_b` | float | Accuracy on Dataset B |
| `rank_a` | int | Rank on Dataset A (1 = worst) |
| `rank_b` | int | Rank on Dataset B (1 = worst) |
| `drift` | int | `rank_a − rank_b` — positive means improved |
| `score_delta` | float | `score_b − score_a` |
| `transfer` | str | `"✓ good"` / `"✗ poor"` / `"~ stable"` |

---

### 6.3 `benchmarks` — Timing Utilities

#### `timer.py`

**`@timed` decorator**

```python
@timed
def my_function(x):
    """Docstring preserved by functools.wraps."""
    ...

my_function(42)
# [timed] my_function took 4.72 ms
```

Uses `functools.wraps` to preserve `__name__`, `__doc__`, and `__module__` of the wrapped function — so introspection tools (like Jupyter's `?` help) still work correctly.

**`benchmark(fn, *args, repeats=5, **kwargs) → float`**

```python
mean_ms = benchmark(rebuild_naive, registry, eval_fn, dataset, repeats=5)
```

Runs the function `repeats` times and returns the **mean elapsed time in milliseconds**. The mean over multiple runs filters out OS scheduling noise, giving a more stable estimate than a single measurement. Uses `time.perf_counter()` for sub-microsecond resolution.

---

## 7. Installation

### Prerequisites

- Python 3.9 or higher
- Git

### Steps

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd capstone_project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install all three packages in editable mode
pip install -e .

# 5. Download and clean both datasets
python data/download.py

# 6. Launch the notebook
jupyter notebook notebook/capstone.ipynb
```

### Verifying the installation

```python
# In a Python shell or notebook cell:
from bst_toolkit import HyperparamRegistry
from ml_toolkit import grid_search, analyse_transfer
from benchmarks import timed, benchmark

reg = HyperparamRegistry()
reg.add_trial(0.95, {"n_estimators": 100})
print(reg.best())
# TrialNode(score=0.9500, params={'n_estimators': 100})
```

### Fallback for UCI dataset download

If the UCI URLs are temporarily unavailable, the `ucimlrepo` package can be used as an alternative:

```bash
pip install ucimlrepo
```

```python
from ucimlrepo import fetch_ucirepo
wdbc     = fetch_ucirepo(id=17)   # Breast Cancer Wisconsin
banknote = fetch_ucirepo(id=267)  # Banknote Authentication
```

Document any fallback used in the notebook's Introduction section.

---

## 8. Usage Guide

### Running a grid search

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from ml_toolkit import grid_search

def evaluate(params, dataset):
    X, y = dataset
    model = RandomForestClassifier(**params, random_state=42)
    scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    return round(float(np.mean(scores)), 6)

param_grid = {
    "n_estimators":    [50, 100, 200],
    "max_depth":       [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
}

registry_A = grid_search(param_grid, evaluate, (X_wdbc, y_wdbc))
# Grid search: 100%|████████████| 36/36 [best=0.9685]
```

### Querying the registry

```python
# Best and worst configurations
print(registry_A.best())
print(registry_A.worst())

# Top 5 configurations
for node in registry_A.top_k(5):
    print(node)

# All configs with accuracy between 94% and 96%
high_accuracy = registry_A.range_query(0.94, 0.96)

# Registry statistics
print(registry_A.summary())
# {'count': 36, 'best_score': 0.9685, 'worst_score': 0.9123,
#  'mean_score': 0.9451, 'tree_height': 7, 'is_balanced': False}
```

### Running Phase 2 — transfer analysis

```python
from bst_toolkit.rebuild import rebuild_balanced
from ml_toolkit import analyse_transfer

# Re-evaluate all configs on Dataset B
registry_B = rebuild_balanced(registry_A, evaluate, (X_banknote, y_banknote))

# Compare rankings
report = analyse_transfer(registry_A, registry_B, drift_threshold=3)

# Top 3 improvers
for entry in report[:3]:
    print(f"drift={entry['drift']:+d}  {entry['transfer']}  {entry['params']}")
```

### Benchmarking rebuild strategies

```python
from benchmarks import benchmark
from bst_toolkit.rebuild import rebuild_naive, rebuild_shuffled, rebuild_balanced

strategies = {
    "naive":    rebuild_naive,
    "shuffled": rebuild_shuffled,
    "balanced": rebuild_balanced,
}

results = {}
for name, fn in strategies.items():
    ms = benchmark(fn, registry_A, evaluate, (X_banknote, y_banknote), repeats=3)
    results[name] = ms
    print(f"{name:10s}: {ms:.1f} ms")
```

---

## 9. How the Two Phases Connect

```
Phase 1                              Phase 2
─────────────────────────────────    ────────────────────────────────────────
Dataset A (wdbc.csv)                 Dataset B (banknote.csv)
    │                                    │
    ▼                                    │
grid_search()                            │
    │                                    │
    ▼                                    │
registry_A  ◄── BST keyed by            │
(36 trials)     score_A                 │
    │                                    │
    └──────────────────┐                 │
                       ▼                 ▼
                  rebuild_balanced(registry_A, evaluate_fn, dataset_B)
                       │
                       ▼
                  registry_B  ◄── BST keyed by score_B
                  (same 36 configs, new scores)
                       │
                       ▼
                  analyse_transfer(registry_A, registry_B)
                       │
                       ▼
                  report  ← sorted by rank drift
                  [✓ good, ~ stable, ✗ poor per config]
```

The key insight is that **the same set of hyperparameter configurations** is evaluated on both datasets. The transfer report answers: *does the ranking of configurations survive the domain change?* A config with high drift (large positive `rank_a − rank_b`) was mediocre on Dataset A but became a top performer on Dataset B — a genuine transfer win.

---

## 10. Algorithm Complexity Summary

| Operation | Average | Worst Case | Where |
|---|---|---|---|
| BST insert | O(log n) | O(n) degenerate | `bst.py` |
| BST delete | O(log n) | O(n) degenerate | `bst.py` |
| BST search | O(log n) | O(n) degenerate | `bst.py` |
| find_min / find_max | O(log n) | O(n) | `bst.py` |
| in-order traversal | O(n) | O(n) | `bst.py` |
| top_k | O(k + h) | O(k + n) | `registry.py` |
| range_query | O(k + h) | O(k + n) | `registry.py` |
| grid_search (n trials) | O(n log n) | O(n²) naive | `grid_search.py` |
| rebuild_naive | O(n²) | O(n²) | `rebuild.py` |
| rebuild_shuffled | O(n log n) expected | O(n²) unlucky | `rebuild.py` |
| rebuild_balanced | O(n log n) | O(n log n) | `rebuild.py` |
| analyse_transfer | O(n log n) | O(n log n) | `transfer.py` |

*n = number of trials stored; h = tree height; k = number of results returned.*

**The degenerate case explained:**  
A BST degenerates into a linked list when nodes are inserted in sorted order. Since `all_trials()` returns nodes in ascending score order, `rebuild_naive` will always insert into an already-sorted sequence, producing a right-skewed tree of height n. This is why `rebuild_naive` is O(n²) while `rebuild_balanced` is O(n log n) — and the benchmark section of the notebook visualises this difference directly.

---

## 11. Grading Alignment

| Criterion | Points | Implementation |
|---|---|---|
| `bst_toolkit` completeness | 25 | All methods in `node.py`, `bst.py`, `registry.py`, `rebuild.py` fully implemented |
| `ml_toolkit` completeness | 15 | `grid_search.py` and `transfer.py` fully implemented |
| `benchmarks` completeness | 5 | `@timed` and `benchmark()` in `timer.py` |
| Git history | 10 | At least 1 commit per package, `git tag v1.0` |
| Notebook structure | 10 | 8 required sections with correct Markdown hierarchy |
| Notebook explanations | 15 | Each result cell followed by interpretation |
| Benchmark analysis | 10 | 3 rebuild strategies timed, degenerate case explained with chart |
| Transfer analysis | 10 | Full report, top-3 improvers/decliners, data science interpretation |
| **Total** | **100** | |

---

## References

- Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
- Lohweg, V. (2013). *Banknote Authentication* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C55P57
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. — Chapters 12–13 (Binary Search Trees)
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.

---

*Generated for Algorithmic Workshop — AIS / EPITA. Authors: Amjad Bsat & Majd Hamoud.*
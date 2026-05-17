# BST-Backed Hyperparameter Optimiser with Transfer Analysis

## Algorithmic Workshop — Final Project

This project implements a complete Python-based data science tool that combines machine learning experimentation with core algorithmic data structures.

The main idea is to run a hyperparameter grid search, store every model trial inside a **Binary Search Tree (BST)** keyed by accuracy score, then re-evaluate the same configurations on a second dataset to analyse how well the configurations transfer.

The project was developed for the **Algorithmic Workshop** course and focuses on connecting practical machine learning workflows with algorithmic concepts such as exhaustive search, BST operations, traversals, divide and conquer, hashing, and benchmarking.

---

## Project Overview

The project has two main phases:

### Phase 1 — Grid Search on Dataset A

A Random Forest model is evaluated on the **Breast Cancer Wisconsin (Diagnostic)** dataset using an exhaustive hyperparameter grid search.

Each trial produces:

- an accuracy score
- a hyperparameter configuration

The score and configuration are stored in a `HyperparamRegistry`, which internally uses a Binary Search Tree.

### Phase 2 — Transfer Analysis on Dataset B

The same hyperparameter configurations are re-evaluated on the **Banknote Authentication** dataset.

The BST is rebuilt using three strategies:

- `rebuild_naive`
- `rebuild_shuffled`
- `rebuild_balanced`

The rankings from Dataset A and Dataset B are then compared to identify which configurations improved, degraded, or stayed stable.

---

## Project Structure

```text
Algorith_Project/
│
├── data/
│   ├── download.py
│   ├── wdbc.csv
│   └── banknote.csv
│
├── bst_toolkit/
│   ├── __init__.py
│   ├── node.py
│   ├── bst.py
│   ├── registry.py
│   └── rebuild.py
│
├── ml_toolkit/
│   ├── __init__.py
│   ├── grid_search.py
│   └── transfer.py
│
├── benchmarks/
│   ├── __init__.py
│   └── timer.py
│
├── notebook/
│   ├── capstone.ipynb
│   └── capstone.html
│
├── setup.py
├── requirements.txt
├── .gitignore
└── README.md

Datasets

This project uses two public datasets from the UCI Machine Learning Repository.

Dataset A — Breast Cancer Wisconsin Diagnostic
File generated: data/wdbc.csv
Samples: 569
Features: 30 numeric features
Target:
0 = benign
1 = malignant

This dataset is used for the initial hyperparameter grid search.

Dataset B — Banknote Authentication
File generated: data/banknote.csv
Samples: 1,372
Features: 4 numeric features
Target:
0 = forged
1 = genuine

This dataset is used to test how well the configurations from Dataset A transfer to a different classification problem.

Data Preparation

The script data/download.py downloads, extracts, preprocesses, and saves both datasets.

It performs the following steps:

Downloads the ZIP files from UCI.
Extracts the raw data files.
Loads the datasets into pandas DataFrames.
Assigns explicit column names.
Encodes target labels.
Standardises feature columns using StandardScaler.
Saves cleaned CSV files.

Run:

python data/download.py

The script is idempotent, meaning it avoids re-downloading or re-extracting files that already exist.

Package 1 — bst_toolkit

This package contains the algorithmic core of the project.

File	Purpose
node.py	Defines the TrialNode dataclass
bst.py	Implements the Binary Search Tree
registry.py	Provides a high-level interface for storing hyperparameter trials
rebuild.py	Implements different BST rebuild strategies

The BST supports:

insertion
deletion
search
find minimum
find maximum
height calculation
balance checking
inorder traversal
preorder traversal
postorder traversal
level-order traversal

The BST is keyed by trial score.

Package 2 — ml_toolkit

This package connects the machine learning workflow to the BST-backed registry.

grid_search.py

Runs an exhaustive hyperparameter grid search.

It:

Receives a parameter grid.
Generates every parameter combination using itertools.product.
Evaluates each configuration using a provided evaluation function.
Stores each score and parameter set inside HyperparamRegistry.
transfer.py

Compares the rankings of the same configurations between two registries.

The output includes:

parameters
score on Dataset A
score on Dataset B
rank on Dataset A
rank on Dataset B
rank drift
transfer label

A positive drift means the configuration improved on Dataset B.

Package 3 — benchmarks

This package provides timing utilities.

Function	Purpose
timed	Decorator that prints how long a function takes
benchmark	Runs a function multiple times and returns the mean elapsed time in milliseconds

These utilities are used in the notebook to compare the three BST rebuild strategies.

Installation

Install the required dependencies:

pip install -r requirements.txt

Install the project locally in editable mode:

pip install -e .

This allows the notebook to import the project packages directly:

from bst_toolkit import HyperparamRegistry
from ml_toolkit import grid_search, analyse_transfer
from benchmarks import benchmark
Running the Project
1. Prepare the datasets
python data/download.py
2. Launch Jupyter Notebook
jupyter notebook

Open:

notebook/capstone.ipynb
3. Run the notebook

In Jupyter Notebook:

Kernel → Restart & Run All

The notebook runs the full project pipeline:

Dataset A exploration
Grid search on Dataset A
BST introspection
Dataset B exploration
BST rebuild benchmarking
Transfer analysis
Final conclusions
Notebook Deliverables

The final notebook is available in:

notebook/capstone.ipynb

An HTML export is also provided:

notebook/capstone.html

The notebook includes:

Markdown explanations
dataset exploration
grid search results
BST traversal demonstrations
benchmark tables
transfer analysis tables
observations and conclusions
Algorithmic Concepts Used
Concept	Where it appears
Exhaustive search	grid_search() tests all hyperparameter combinations
Binary Search Tree	Scores are stored as BST keys
Recursion	BST insertion, deletion, traversal, and rebuilding
Inorder traversal	Retrieves scores in sorted order
Reverse inorder traversal	Retrieves top-k configurations
Range query pruning	Retrieves scores within a selected interval
Divide and conquer	Builds a balanced BST from sorted scores
Hash tables	Used in transfer ranking lookup
Benchmarking	Compares rebuild strategies
Cross-validation	Evaluates Random Forest configurations
Main Results

The grid search on Dataset A found several Random Forest configurations with similar accuracy. The best stored configuration achieved an accuracy of approximately 0.963.

After re-evaluating the same configurations on Dataset B, the best score increased to approximately 0.996. The transfer analysis showed that some configurations improved significantly, while others dropped in ranking.

This demonstrates that strong hyperparameters on one dataset may transfer well, but their relative ranking can still change depending on the dataset structure.

Final Submission Checklist

Before submission, the following checks were completed:

All packages are importable.
Dataset CSV files are generated.
grid_search() runs successfully.
analyse_transfer() generates a transfer report.
benchmark() measures rebuild strategy performance.
capstone.ipynb runs from top to bottom.
capstone.html is exported.
Git history contains meaningful commits.
Final project tag is created.
Authors

This project was completed as part of the Algorithmic Workshop final project.

Contributors worked on different parts of the system, including:

BST implementation
dataset preparation
grid search
transfer analysis
benchmarking
final notebook and documentation
Notes on AI Assistance

AI tools were used as support for debugging, explanation, and code review. The project implementation was tested step by step, and the final notebook explains the code and algorithmic reasoning behind each part.


After pasting and saving, run:

```bash
git status
git add README.md
git commit -m "docs: improve project README"
git push origin main
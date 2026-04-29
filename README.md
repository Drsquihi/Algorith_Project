# Algorith_Project
BST-based hyperparameter optimization framework with transfer learning analysis across datasets, built using custom data structures and machine learning pipelines.

# BST-Backed Hyperparameter Optimiser with Transfer Analysis

This project is a complete data science and algorithmic system that combines custom data structures with machine learning workflows.

It implements a Binary Search Tree (BST) to efficiently store, rank, and analyse hyperparameter configurations based on model performance.

## 🚀 Project Overview

The project is divided into two main phases:

### Phase 1 — Hyperparameter Optimisation
- Perform an exhaustive grid search on Dataset A
- Evaluate each configuration using cross-validation
- Store results in a BST keyed by accuracy score
- Enable efficient queries such as top-k results and score ranges

### Phase 2 — Transfer Analysis
- Re-evaluate the same configurations on Dataset B
- Rebuild the BST using three strategies:
  - Naive (sorted insertion)
  - Shuffled insertion
  - Balanced reconstruction
- Analyse how well configurations transfer across datasets

## 🧠 Key Concepts

- Binary Search Trees (BST)
- Recursion and tree traversal algorithms
- Divide & Conquer (balanced tree reconstruction)
- Time complexity analysis (O(n), O(log n), O(n²))
- Grid search and model evaluation
- Transfer learning intuition

## 📦 Project Structure

- `bst_toolkit/` → Core BST implementation and registry system
- `ml_toolkit/` → Grid search and transfer analysis
- `benchmarks/` → Performance measurement tools
- `data/` → Dataset download and preprocessing scripts
- `notebook/` → Final analysis and results (Jupyter Notebook)

## 📊 Datasets

- Breast Cancer Wisconsin (Diagnostic)
- Banknote Authentication

These datasets are automatically downloaded and preprocessed using a custom data pipeline.

## ⚙️ Installation

```bash
pip install -r requirements.txt
pip install -e .

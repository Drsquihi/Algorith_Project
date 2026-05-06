"""
data/download.py
----------------
Downloads, extracts, and cleans two UCI datasets:
  - Dataset A: Breast Cancer Wisconsin (Diagnostic) → data/wdbc.csv
  - Dataset B: Banknote Authentication             → data/banknote.csv

Idempotent: skips download/extraction if files already exist.
Usage:
    python data/download.py
"""

import os
import io
import zipfile
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Configuration ────────────────────────────────────────────────────────────

DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # same folder as this script

DATASETS = {
    "wdbc": {
        "url": "https://archive.ics.uci.edu/static/public/17/breast+cancer+wisconsin+diagnostic.zip",
        "zip_name": "wdbc.zip",
        "data_file": "wdbc.data",
        "csv_out": "wdbc.csv",
    },
    "banknote": {
        "url": "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip",
        "zip_name": "banknote.zip",
        "data_file": "data_banknote_authentication.txt",
        "csv_out": "banknote.csv",
    },
}

# ── Column definitions ────────────────────────────────────────────────────────

# wdbc.data: id, diagnosis, then 30 features (mean / se / worst for 10 measurements)
_MEASUREMENTS = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension",
]
WDBC_COLUMNS = (
    ["id", "diagnosis"]
    + [f"{m}_mean"  for m in _MEASUREMENTS]
    + [f"{m}_se"    for m in _MEASUREMENTS]
    + [f"{m}_worst" for m in _MEASUREMENTS]
)  # 32 columns total

# banknote: 4 features + 1 label
BANKNOTE_COLUMNS = ["variance", "skewness", "curtosis", "entropy", "label"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _download_zip(url: str, dest_path: str) -> None:
    """Download a ZIP file from *url* and save it to *dest_path*."""
    print(f"  Downloading {url} …")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)
    print(f"  Saved to {dest_path}")


def _extract_file(zip_path: str, filename: str, dest_dir: str) -> str:
    """Extract *filename* from *zip_path* into *dest_dir*; return the output path."""
    out_path = os.path.join(dest_dir, filename)
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Some ZIPs nest files inside sub-folders; find the entry by name suffix
        matches = [n for n in zf.namelist() if n.endswith(filename)]
        if not matches:
            raise FileNotFoundError(
                f"'{filename}' not found in {zip_path}. "
                f"Available entries: {zf.namelist()}"
            )
        entry = matches[0]
        with zf.open(entry) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
    print(f"  Extracted '{filename}' → {out_path}")
    return out_path


# ── Dataset-specific loaders ──────────────────────────────────────────────────

def load_wdbc(data_path: str) -> pd.DataFrame:
    """
    Load wdbc.data, drop 'id', encode 'diagnosis' (M→1, B→0),
    scale the 30 feature columns with StandardScaler.
    Returns a DataFrame with columns: [feature_0 … feature_29, target].
    """
    df = pd.read_csv(data_path, header=None, names=WDBC_COLUMNS)

    # Drop the ID column
    df.drop(columns=["id"], inplace=True)

    # Encode target: M → 1, B → 0
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0}).astype(int)
    df.rename(columns={"diagnosis": "target"}, inplace=True)

    # Scale features
    feature_cols = [c for c in df.columns if c != "target"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df


def load_banknote(data_path: str) -> pd.DataFrame:
    """
    Load banknote data, assign column names, scale the 4 feature columns.
    Target column ('label') is already 0/1 integers.
    """
    df = pd.read_csv(data_path, header=None, names=BANKNOTE_COLUMNS)

    df["label"] = df["label"].astype(int)
    df.rename(columns={"label": "target"}, inplace=True)

    # Scale features
    feature_cols = [c for c in df.columns if c != "target"]
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    return df


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_dataset(key: str, loader_fn) -> None:
    cfg = DATASETS[key]
    zip_path  = os.path.join(DATA_DIR, cfg["zip_name"])
    data_path = os.path.join(DATA_DIR, cfg["data_file"])
    csv_path  = os.path.join(DATA_DIR, cfg["csv_out"])

    # 1. Download (idempotent)
    if not os.path.exists(zip_path):
        _download_zip(cfg["url"], zip_path)
    else:
        print(f"  ZIP already present: {zip_path}")

    # 2. Extract (idempotent)
    if not os.path.exists(data_path):
        _extract_file(zip_path, cfg["data_file"], DATA_DIR)
    else:
        print(f"  Data file already present: {data_path}")

    # 3. Load, preprocess, save CSV
    df = loader_fn(data_path)
    df.to_csv(csv_path, index=False)

    print(f"  ✓ {cfg['csv_out']} saved — {len(df)} rows, {df.shape[1]} columns "
          f"(target distribution: {df['target'].value_counts().to_dict()})\n")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=== Dataset A: Breast Cancer Wisconsin (Diagnostic) ===")
    process_dataset("wdbc", load_wdbc)

    print("=== Dataset B: Banknote Authentication ===")
    process_dataset("banknote", load_banknote)

    print("All datasets ready.")


if __name__ == "__main__":
    main()
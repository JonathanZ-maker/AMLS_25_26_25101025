"""
AMLS Assignment - Main entry point.

Step 1 objective:
- Run Model A (Kernel SVM) raw-flatten baseline on BreastMNIST.
- Write a single row to results/summary.csv.

Run:
    python main.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd

from model_A.data import load_breastmnist
from model_A.features import flatten_images, fit_standardizer, apply_standardizer
from model_A.train_eval import SVMBaselineConfig, train_svm_baseline, eval_binary_classification


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_results_dir(out_dir: Path) -> None:
    """Create output directories if missing."""
    out_dir.mkdir(parents=True, exist_ok=True)


def append_row_to_csv(csv_path: Path, row: dict) -> None:
    """
    Append one experiment row to a CSV file. Creates file if not exists.
    """
    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(csv_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    set_seed(args.seed)
    project_root = Path(__file__).resolve().parent
    out_dir = project_root / args.out_dir
    ensure_results_dir(out_dir)

    # ---------- Load data ----------
    splits = load_breastmnist(project_root)

    # Normalize images to [0, 1] (common for uint8 inputs)
    x_train = splits.x_train.astype(np.float32) / 255.0
    y_train = splits.y_train
    x_val = splits.x_val.astype(np.float32) / 255.0
    y_val = splits.y_val
    x_test = splits.x_test.astype(np.float32) / 255.0
    y_test = splits.y_test

    # ---------- Feature extraction (Raw flatten) ----------
    xtr = flatten_images(x_train)
    xva = flatten_images(x_val)
    xte = flatten_images(x_test)

    # Standardize based on training data only
    scaler = fit_standardizer(xtr)
    xtr = apply_standardizer(scaler, xtr)
    xva = apply_standardizer(scaler, xva)
    xte = apply_standardizer(scaler, xte)

    # ---------- Train baseline SVM ----------
    cfg = SVMBaselineConfig(C=1.0, gamma=0.1)
    model = train_svm_baseline(xtr, y_train, cfg)

    # Evaluate on test (Step 1 requirement)
    y_pred_test = model.predict(xte)
    metrics_test = eval_binary_classification(y_test, y_pred_test)

    # ---------- Save results ----------
    summary_path = out_dir / "summary.csv"
    row = {
        "model": "A",
        "pipeline": "raw_flatten",
        "augmentation": "NoAug",
        "capacity": f"C={cfg.C},gamma={cfg.gamma}",
        "budget": "data=100%",
        "split": "test",
        **metrics_test,
        "seed": args.seed,
    }
    append_row_to_csv(summary_path, row)

    print("Step 1 finished. Test metrics:")
    for k, v in metrics_test.items():
        print(f"  {k}: {v:.4f}")
    print(f"Saved to: {summary_path}")


if __name__ == "__main__":
    main()

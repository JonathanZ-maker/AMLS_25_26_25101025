"""
AMLS Assignment - Main entry point (Step 2: Model A).

This main.py runs a robust, report-aligned experiment matrix for Model A:

A1 Baselines:
  - raw_flatten + RBF SVM (grid search on val: C x gamma)
  - HOG + Linear SVM (grid search on val: C only)

A2 Capacity:
  - raw_flatten + RBF: sweep all (C, gamma) on test
  - HOG + Linear: sweep C on test

A3 Augmentation sensitivity (HOG + Linear only):
  - NoAug / RotShift / RotShift+Noise

A4 Training budget (HOG + Linear only):
  - 25% / 50% / 100% training data

Outputs:
- results/summary.csv
- results/plots/*.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model_A.data import load_breastmnist
from model_A.features import flatten_images, hog_features, fit_standardizer, apply_standardizer
from model_A.augment import random_rotate_shift, add_gaussian_noise
from model_A.train_eval import (
    SVMGrid,
    SVMTrainConfig,
    grid_search_svm,
    train_and_eval,
)


def set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def ensure_dirs(out_dir: Path) -> None:
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)


def append_row(csv_path: Path, row: dict) -> None:
    """
    Append a row to CSV, robust to empty/corrupted files.
    """
    df_new = pd.DataFrame([row])
    if csv_path.exists():
        try:
            if csv_path.stat().st_size == 0:
                df_new.to_csv(csv_path, index=False)
                return
            df_old = pd.read_csv(csv_path)
            if df_old.shape[1] == 0:
                df_new.to_csv(csv_path, index=False)
                return
            df_all = pd.concat([df_old, df_new], ignore_index=True)
            df_all.to_csv(csv_path, index=False)
            return
        except Exception:
            df_new.to_csv(csv_path, index=False)
            return
    df_new.to_csv(csv_path, index=False)


def subsample(x: np.ndarray, y: np.ndarray, frac: float, rng: np.random.Generator):
    """
    Subsample training data by fraction with per-class sampling.
    """
    if frac >= 0.999:
        return x, y

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n0 = max(1, int(len(idx0) * frac))
    n1 = max(1, int(len(idx1) * frac))
    sel0 = rng.choice(idx0, size=n0, replace=False)
    sel1 = rng.choice(idx1, size=n1, replace=False)
    sel = np.concatenate([sel0, sel1])
    rng.shuffle(sel)
    return x[sel], y[sel]


def build_features(pipeline: str, x: np.ndarray) -> np.ndarray:
    if pipeline == "raw_flatten":
        return flatten_images(x)
    if pipeline == "hog":
        return hog_features(x)
    raise ValueError(f"Unknown pipeline: {pipeline}")


def standardize_by_train(f_tr: np.ndarray, f_va: np.ndarray, f_te: np.ndarray):
    scaler = fit_standardizer(f_tr)
    return (
        apply_standardizer(scaler, f_tr),
        apply_standardizer(scaler, f_va),
        apply_standardizer(scaler, f_te),
    )


def plot_capacity(df: pd.DataFrame, out_dir: Path) -> None:
    """
    Generate capacity plots for:
    - raw + RBF: heatmap over (C, gamma)
    - hog + linear: line plot over C
    """
    plots_dir = out_dir / "plots"

    # raw + rbf heatmap
    cap_raw = df[(df["model"] == "A") & (df["experiment"] == "A2_capacity_raw_rbf")].copy()
    if not cap_raw.empty:
        pivot = cap_raw.pivot_table(index="capacity_gamma", columns="capacity_C", values="f1", aggfunc="mean")
        mat = pivot.values
        gammas = pivot.index.values
        Cs = pivot.columns.values

        plt.figure()
        plt.imshow(mat, aspect="auto", origin="lower")
        plt.xticks(ticks=np.arange(len(Cs)), labels=[str(c) for c in Cs])
        plt.yticks(ticks=np.arange(len(gammas)), labels=[str(g) for g in gammas])
        plt.xlabel("C")
        plt.ylabel("gamma")
        plt.title("Model A Capacity (Raw + RBF): F1 on Test")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(plots_dir / "fig_A_capacity_raw_rbf.png", dpi=200)
        plt.close()

    # hog + linear line
    cap_hog = df[(df["model"] == "A") & (df["experiment"] == "A2_capacity_hog_linear")].copy()
    if not cap_hog.empty:
        cap_hog = cap_hog.sort_values("capacity_C")
        plt.figure()
        plt.plot(cap_hog["capacity_C"].astype(float), cap_hog["f1"].astype(float), marker="o")
        plt.xlabel("C")
        plt.ylabel("F1 (test)")
        plt.title("Model A Capacity (HOG + Linear): F1 on Test")
        plt.tight_layout()
        plt.savefig(plots_dir / "fig_A_capacity_hog_linear.png", dpi=200)
        plt.close()


def plot_aug(df: pd.DataFrame, out_dir: Path) -> None:
    aug = df[(df["model"] == "A") & (df["experiment"] == "A3_augmentation_hog_linear")].copy()
    if aug.empty:
        return

    order = ["NoAug", "RotShift", "RotShift+Noise"]
    aug["augmentation"] = pd.Categorical(aug["augmentation"], categories=order, ordered=True)
    aug = aug.sort_values("augmentation")

    plt.figure()
    plt.bar(aug["augmentation"].astype(str), aug["f1"].astype(float))
    plt.xlabel("Augmentation")
    plt.ylabel("F1 (test)")
    plt.title("Model A Augmentation (HOG + Linear)")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "fig_A_aug_hog_linear.png", dpi=200)
    plt.close()


def plot_budget(df: pd.DataFrame, out_dir: Path) -> None:
    bud = df[(df["model"] == "A") & (df["experiment"] == "A4_budget_hog_linear")].copy()
    if bud.empty:
        return
    bud = bud.sort_values("budget_data_frac")

    plt.figure()
    plt.plot(bud["budget_data_frac"].astype(float), bud["f1"].astype(float), marker="o")
    plt.xlabel("Training data fraction")
    plt.ylabel("F1 (test)")
    plt.title("Model A Data Budget (HOG + Linear)")
    plt.tight_layout()
    plt.savefig(out_dir / "plots" / "fig_A_budget_hog_linear.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--reset", action="store_true", help="Reset results/summary.csv before running.")
    args = parser.parse_args()

    rng = set_seed(args.seed)
    root = Path(__file__).resolve().parent
    out_dir = root / args.out_dir
    ensure_dirs(out_dir)
    summary_csv = out_dir / "summary.csv"

    if args.reset and summary_csv.exists():
        summary_csv.unlink()

    # Load dataset
    splits = load_breastmnist(root)
    x_train = splits.x_train.astype(np.float32) / 255.0
    y_train = splits.y_train.astype(np.int64)
    x_val = splits.x_val.astype(np.float32) / 255.0
    y_val = splits.y_val.astype(np.int64)
    x_test = splits.x_test.astype(np.float32) / 255.0
    y_test = splits.y_test.astype(np.int64)

    grid = SVMGrid()

    # ==========================================================
    # A1: Baselines
    # ==========================================================
    # A1-raw: RBF SVM (C x gamma)
    raw_cfg = SVMTrainConfig(kernel="rbf", class_weight="balanced")
    f_tr = build_features("raw_flatten", x_train)
    f_va = build_features("raw_flatten", x_val)
    f_te = build_features("raw_flatten", x_test)
    f_tr, f_va, f_te = standardize_by_train(f_tr, f_va, f_te)

    best_C, best_gamma, best_val_metrics, _ = grid_search_svm(
        f_tr, y_train, f_va, y_val, grid=grid, cfg=raw_cfg, selection_metric="f1_macro"
    )
    out = train_and_eval(f_tr, y_train, f_te, y_test, C=best_C, gamma=best_gamma, cfg=raw_cfg)
    append_row(summary_csv, {
        "model": "A",
        "experiment": "A1_baseline_raw_rbf",
        "pipeline": "raw_flatten",
        "augmentation": "NoAug",
        "kernel": "rbf",
        "capacity": f"C={best_C},gamma={best_gamma}",
        "capacity_C": float(best_C),
        "capacity_gamma": float(best_gamma) if best_gamma is not None else np.nan,
        "budget": "data=100%",
        "budget_data_frac": 1.0,
        "split": "test",
        **out,
        "seed": args.seed,
    })

    # A1-hog: Linear SVM (C only)
    hog_cfg = SVMTrainConfig(kernel="linear", class_weight="balanced")
    f_tr = build_features("hog", x_train)
    f_va = build_features("hog", x_val)
    f_te = build_features("hog", x_test)
    f_tr, f_va, f_te = standardize_by_train(f_tr, f_va, f_te)

    best_C, best_gamma, best_val_metrics, _ = grid_search_svm(
        f_tr, y_train, f_va, y_val, grid=grid, cfg=hog_cfg, selection_metric="f1_macro"
    )
    out = train_and_eval(f_tr, y_train, f_te, y_test, C=best_C, gamma=None, cfg=hog_cfg)
    append_row(summary_csv, {
        "model": "A",
        "experiment": "A1_baseline_hog_linear",
        "pipeline": "hog",
        "augmentation": "NoAug",
        "kernel": "linear",
        "capacity": f"C={best_C}",
        "capacity_C": float(best_C),
        "capacity_gamma": np.nan,
        "budget": "data=100%",
        "budget_data_frac": 1.0,
        "split": "test",
        **out,
        "seed": args.seed,
    })

    # ==========================================================
    # A2: Capacity
    # ==========================================================
    # A2-raw-rbf: sweep C x gamma
    f_tr = build_features("raw_flatten", x_train)
    f_va = build_features("raw_flatten", x_val)
    f_te = build_features("raw_flatten", x_test)
    f_tr, f_va, f_te = standardize_by_train(f_tr, f_va, f_te)

    for C in grid.C_list:
        for gamma in grid.gamma_list:
            out = train_and_eval(f_tr, y_train, f_te, y_test, C=float(C), gamma=float(gamma), cfg=raw_cfg)
            append_row(summary_csv, {
                "model": "A",
                "experiment": "A2_capacity_raw_rbf",
                "pipeline": "raw_flatten",
                "augmentation": "NoAug",
                "kernel": "rbf",
                "capacity": f"C={C},gamma={gamma}",
                "capacity_C": float(C),
                "capacity_gamma": float(gamma),
                "budget": "data=100%",
                "budget_data_frac": 1.0,
                "split": "test",
                **out,
                "seed": args.seed,
            })

    # A2-hog-linear: sweep C only
    f_tr = build_features("hog", x_train)
    f_va = build_features("hog", x_val)
    f_te = build_features("hog", x_test)
    f_tr, f_va, f_te = standardize_by_train(f_tr, f_va, f_te)

    for C in grid.C_list:
        out = train_and_eval(f_tr, y_train, f_te, y_test, C=float(C), gamma=None, cfg=hog_cfg)
        append_row(summary_csv, {
            "model": "A",
            "experiment": "A2_capacity_hog_linear",
            "pipeline": "hog",
            "augmentation": "NoAug",
            "kernel": "linear",
            "capacity": f"C={C}",
            "capacity_C": float(C),
            "capacity_gamma": np.nan,
            "budget": "data=100%",
            "budget_data_frac": 1.0,
            "split": "test",
            **out,
            "seed": args.seed,
        })

    # ==========================================================
    # A3: Augmentation sensitivity (HOG + Linear only)
    # ==========================================================
    # Select best C on unaugmented data as reference
    f_tr0 = build_features("hog", x_train)
    f_va0 = build_features("hog", x_val)
    f_te0 = build_features("hog", x_test)
    f_tr0, f_va0, f_te0 = standardize_by_train(f_tr0, f_va0, f_te0)

    best_C, _, _, _ = grid_search_svm(
        f_tr0, y_train, f_va0, y_val, grid=grid, cfg=hog_cfg, selection_metric="f1_macro"
    )

    aug_settings = [
        ("NoAug", None),
        ("RotShift", "rotshift"),
        ("RotShift+Noise", "rotshift_noise"),
    ]

    for aug_name, aug_code in aug_settings:
        x_tr_aug = x_train.copy()
        if aug_code == "rotshift":
            x_tr_aug = random_rotate_shift(x_tr_aug, rng)
        elif aug_code == "rotshift_noise":
            x_tr_aug = random_rotate_shift(x_tr_aug, rng)
            x_tr_aug = add_gaussian_noise(x_tr_aug, rng)

        f_tr = build_features("hog", x_tr_aug)
        f_va = build_features("hog", x_val)
        f_te = build_features("hog", x_test)
        f_tr, f_va, f_te = standardize_by_train(f_tr, f_va, f_te)

        out = train_and_eval(f_tr, y_train, f_te, y_test, C=float(best_C), gamma=None, cfg=hog_cfg)
        append_row(summary_csv, {
            "model": "A",
            "experiment": "A3_augmentation_hog_linear",
            "pipeline": "hog",
            "augmentation": aug_name,
            "kernel": "linear",
            "capacity": f"C={best_C}",
            "capacity_C": float(best_C),
            "capacity_gamma": np.nan,
            "budget": "data=100%",
            "budget_data_frac": 1.0,
            "split": "test",
            **out,
            "seed": args.seed,
        })

    # ==========================================================
    # A4: Training data budget (HOG + Linear only)
    # ==========================================================
    for frac in [0.25, 0.50, 1.00]:
        x_sub, y_sub = subsample(x_train, y_train, frac, rng)

        f_tr = build_features("hog", x_sub)
        f_va = build_features("hog", x_val)
        f_te = build_features("hog", x_test)
        f_tr, f_va, f_te = standardize_by_train(f_tr, f_va, f_te)

        best_C, _, _, _ = grid_search_svm(
            f_tr, y_sub, f_va, y_val, grid=grid, cfg=hog_cfg, selection_metric="f1_macro"
        )
        out = train_and_eval(f_tr, y_sub, f_te, y_test, C=float(best_C), gamma=None, cfg=hog_cfg)

        append_row(summary_csv, {
            "model": "A",
            "experiment": "A4_budget_hog_linear",
            "pipeline": "hog",
            "augmentation": "NoAug",
            "kernel": "linear",
            "capacity": f"C={best_C}",
            "capacity_C": float(best_C),
            "capacity_gamma": np.nan,
            "budget": f"data={int(frac*100)}%",
            "budget_data_frac": float(frac),
            "split": "test",
            **out,
            "seed": args.seed,
        })

    # Plot report figures
    df = pd.read_csv(summary_csv)
    plot_capacity(df, out_dir)
    plot_aug(df, out_dir)
    plot_budget(df, out_dir)

    print("Step 2 finished: Model A experiments completed.")
    print(f"Saved summary to: {summary_csv}")
    print(f"Saved plots to: {out_dir / 'plots'}")


if __name__ == "__main__":
    main()

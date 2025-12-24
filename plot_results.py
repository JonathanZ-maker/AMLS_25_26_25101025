# plot_results.py
# AMLS 25-26: Generate Model B report figures from results/summary.csv
#
# Usage (from project root):
#   python plot_results.py --summary results/summary.csv --outdir results/plots
#
# This script produces *report-ready* figures for Model B (ResNet-18):
#   - fig_B_aug_f1.png          (B2 augmentation sensitivity)
#   - fig_B_epoch_budget_f1.png (B3 epoch budget)
#   - fig_B_data_budget_f1.png  (B4 data budget)
#   - fig_B_capacity_f1.png     (B5 frozen vs full fine-tune)
#
# Optional: export compact LaTeX tables:
#   - table_B_baseline.tex
#   - table_B_capacity.tex

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _require_columns(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"summary.csv missing columns: {missing}")


def _filter_model_b(df: pd.DataFrame) -> pd.DataFrame:
    # Accept both "B" and "b" just in case
    return df[df["model"].astype(str).str.upper() == "B"].copy()


def _save_bar_figure(x_labels, y_values, out_path: Path,
                     title: str, xlabel: str, ylabel: str) -> None:
    plt.figure()
    bars = plt.bar(x_labels, y_values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Annotate each bar with value (3 decimals)
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _save_line_figure(x_values, y_values, out_path: Path,
                      title: str, xlabel: str, ylabel: str) -> None:
    plt.figure()
    plt.plot(x_values, y_values, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # Annotate each point with value (3 decimals)
    for x, y in zip(x_values, y_values):
        plt.text(
            x,
            y,
            f"{y:.3f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()



def _to_percent(frac: float) -> int:
    return int(round(frac * 100))


def export_model_b_figures(summary_csv: Path, outdir: Path, prefer_metric: str = "f1") -> None:
    df = pd.read_csv(summary_csv)

    required = [
        "model", "experiment", "augmentation", "capacity",
        "budget_data_frac", "budget_epochs",
        "accuracy", "precision", "recall", "f1", "bal_acc", "seed"
    ]
    _require_columns(df, required)

    dfb = _filter_model_b(df)
    if dfb.empty:
        raise ValueError("No Model B rows found in summary.csv (model == 'B').")

    # Ensure numeric types
    for c in ["budget_data_frac", "budget_epochs", "accuracy", "precision", "recall", "f1", "bal_acc"]:
        dfb[c] = pd.to_numeric(dfb[c], errors="coerce")

    # ---------- B2: augmentation sensitivity ----------
    # We plot F1 (or selected metric) for NoAug / RotShift / RotShift+Noise at fixed budget/capacity
    df_b2 = dfb[dfb["experiment"].astype(str).str.contains("B2_augmentation", na=False)].copy()
    if not df_b2.empty:
        aug_order = ["NoAug", "RotShift", "RotShift+Noise"]
        df_b2["augmentation"] = df_b2["augmentation"].astype(str)
        df_b2 = df_b2.set_index("augmentation").reindex(aug_order).reset_index()

        y = df_b2[prefer_metric].to_numpy()
        out_path = outdir / "fig_B_aug_f1.png"
        _save_bar_figure(
            x_labels=aug_order,
            y_values=y,
            out_path=out_path,
            title=f"Model B: Augmentation Sensitivity ({prefer_metric.upper()} on test set)",
            xlabel="Augmentation",
            ylabel=prefer_metric.upper(),
        )
    else:
        print("[WARN] No rows for B2_augmentation found. Skipping fig_B_aug_f1.png")

    # ---------- B3: epoch budget ----------
    df_b3 = dfb[dfb["experiment"].astype(str).str.contains("B3_budget_epochs", na=False)].copy()
    if not df_b3.empty:
        df_b3 = df_b3.sort_values("budget_epochs")
        x = df_b3["budget_epochs"].astype(int).to_numpy()
        y = df_b3[prefer_metric].to_numpy()
        out_path = outdir / "fig_B_epoch_budget_f1.png"
        _save_line_figure(
            x_values=x,
            y_values=y,
            out_path=out_path,
            title=f"Model B: Epoch Budget ({prefer_metric.upper()} on test set)",
            xlabel="Epochs",
            ylabel=prefer_metric.upper(),
        )
    else:
        print("[WARN] No rows for B3_budget_epochs found. Skipping fig_B_epoch_budget_f1.png")

    # ---------- B4: data budget ----------
    df_b4 = dfb[dfb["experiment"].astype(str).str.contains("B4_budget_data", na=False)].copy()
    if not df_b4.empty:
        df_b4 = df_b4.sort_values("budget_data_frac")
        x = [_to_percent(v) for v in df_b4["budget_data_frac"].to_numpy()]
        y = df_b4[prefer_metric].to_numpy()
        out_path = outdir / "fig_B_data_budget_f1.png"
        _save_line_figure(
            x_values=x,
            y_values=y,
            out_path=out_path,
            title=f"Model B: Training Data Budget ({prefer_metric.upper()} on test set)",
            xlabel="Training data used (%)",
            ylabel=prefer_metric.upper(),
        )
    else:
        print("[WARN] No rows for B4_budget_data found. Skipping fig_B_data_budget_f1.png")

    # ---------- B5: capacity (frozen vs full) ----------
    df_b5 = dfb[dfb["experiment"].astype(str).str.contains("B5_capacity", na=False)].copy()
    if not df_b5.empty:
        # Order: frozen first, full second
        cap_order = ["frozen_backbone", "full_finetune"]
        df_b5["capacity"] = df_b5["capacity"].astype(str)
        df_b5 = df_b5.set_index("capacity").reindex(cap_order).reset_index()

        labels = ["Frozen", "Full"]
        y = df_b5[prefer_metric].to_numpy()
        out_path = outdir / "fig_B_capacity_f1.png"
        _save_bar_figure(
            x_labels=labels,
            y_values=y,
            out_path=out_path,
            title=f"Model B: Capacity Comparison ({prefer_metric.upper()} on test set)",
            xlabel="Training regime",
            ylabel=prefer_metric.upper(),
        )
    else:
        print("[WARN] No rows for B5_capacity found. Skipping fig_B_capacity_f1.png")

    print(f"[OK] Model B figures saved to: {outdir.resolve()}")


def export_model_b_tables(summary_csv: Path, outdir: Path) -> None:
    """
    Optional convenience: export small LaTeX tables so you can paste into Overleaf quickly.
    If you prefer manual tables, you can ignore this entirely.
    """
    df = pd.read_csv(summary_csv)
    _require_columns(df, ["model", "experiment", "augmentation", "capacity", "budget_data_frac", "budget_epochs",
                          "accuracy", "precision", "recall", "f1", "bal_acc"])

    dfb = _filter_model_b(df)
    if dfb.empty:
        print("[WARN] No Model B rows found, skipping LaTeX tables.")
        return

    outdir.mkdir(parents=True, exist_ok=True)

    # Baseline row (B1)
    b1 = dfb[dfb["experiment"].astype(str).str.contains("B1_baseline", na=False)].copy()
    if not b1.empty:
        b1 = b1.iloc[0]
        tbl = pd.DataFrame([{
            "Aug": str(b1["augmentation"]),
            "Epochs": int(b1["budget_epochs"]),
            "Data(%)": _to_percent(float(b1["budget_data_frac"])),
            "Acc": float(b1["accuracy"]),
            "F1": float(b1["f1"]),
            "BalAcc": float(b1["bal_acc"]),
        }])
        tex_path = outdir / "table_B_baseline.tex"
        tbl.to_latex(tex_path, index=False, float_format="%.3f")
        print(f"[OK] Wrote {tex_path}")
    else:
        print("[WARN] No B1_baseline row found, skipping table_B_baseline.tex")

    # Capacity rows (B5)
    b5 = dfb[dfb["experiment"].astype(str).str.contains("B5_capacity", na=False)].copy()
    if not b5.empty:
        # Map labels
        b5["Regime"] = b5["capacity"].astype(str).map({
            "frozen_backbone": "Frozen",
            "full_finetune": "Full"
        }).fillna(b5["capacity"].astype(str))
        b5 = b5[["Regime", "accuracy", "f1", "bal_acc"]].copy()
        b5 = b5.sort_values("Regime")
        tex_path = outdir / "table_B_capacity.tex"
        b5.to_latex(tex_path, index=False, float_format="%.3f")
        print(f"[OK] Wrote {tex_path}")
    else:
        print("[WARN] No B5_capacity rows found, skipping table_B_capacity.tex")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="results/summary.csv",
                        help="Path to results/summary.csv")
    parser.add_argument("--outdir", type=str, default="results/plots",
                        help="Directory to save figures/tables")
    parser.add_argument("--metric", type=str, default="f1",
                        choices=["f1", "accuracy", "bal_acc", "f1_macro", "precision", "recall"],
                        help="Metric used for Model B result figures")
    parser.add_argument("--export_tables", action="store_true",
                        help="Also export small LaTeX tables for report")
    args = parser.parse_args()

    summary_csv = Path(args.summary)
    outdir = Path(args.outdir)

    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv.resolve()}")

    export_model_b_figures(summary_csv, outdir, prefer_metric=args.metric)

    if args.export_tables:
        export_model_b_tables(summary_csv, outdir)


if __name__ == "__main__":
    main()

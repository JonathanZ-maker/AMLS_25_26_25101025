"""
Model B - Train/Eval utilities for ResNet-18 adapted.

Outputs:
- metrics dict compatible with your summary.csv fields (acc/prec/rec/f1 + extras)
- optional learning curve CSV for convergence reporting in Section 4
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score


@dataclass
class TrainConfigB:
    aug: str = "NoAug"
    data_frac: float = 1.0
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    train_backbone: bool = True  # capacity: frozen vs full
    seed: int = 42
    device: str = "cuda"  # auto-fallback handled outside


def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def _eval(model: nn.Module, loader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        ys.append(y.detach().cpu().numpy())
        ps.append(pred.detach().cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _metrics_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pred_pos_rate = float((y_pred == 1).mean())

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "f1_macro": float(f1_macro),
        "bal_acc": float(bal_acc),
        "pred_pos_rate": float(pred_pos_rate),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def train_and_eval_b(
    
    model: nn.Module,
    dl_tr,
    dl_va,
    dl_te,
    cfg: TrainConfigB,
    out_dir: Path,
) -> Dict[str, float]:
    _set_seed(cfg.seed)

    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = model.to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    curves = []
    best_val_f1 = -1.0
    best_state = None

    for ep in range(1, cfg.epochs + 1):
        model.train()
        tr_losses = []
        for x, y in dl_tr:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            tr_losses.append(loss.item())

        # val metrics each epoch (for convergence reporting)
        yv, pv = _eval(model, dl_va, device)
        val_f1 = f1_score(yv, pv, zero_division=0)

        curves.append({
            "epoch": ep,
            "train_loss": float(np.mean(tr_losses)) if tr_losses else np.nan,
            "val_f1": float(val_f1),
        })
        print(f"[Model B] ep {ep:03d}/{cfg.epochs} | train_loss={np.mean(tr_losses):.4f} | val_f1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # save learning curve (needed by Section 4 convergence discussion)
    curves_dir = out_dir / "curves"
    curves_dir.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df_curve = pd.DataFrame(curves)

    curve_path = curves_dir / (
        f"B_curve_aug={cfg.aug}_data={cfg.data_frac}_ep={cfg.epochs}_"
        f"cap={'full' if cfg.train_backbone else 'frozen'}_seed={cfg.seed}.csv"
    )
    df_curve.to_csv(curve_path, index=False)

    # save learning curve figures (PNG) for report Section 4
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig_path_loss = plots_dir / (
        f"fig_B_trainloss_aug={cfg.aug}_data={cfg.data_frac}_ep={cfg.epochs}_"
        f"cap={'full' if cfg.train_backbone else 'frozen'}_seed={cfg.seed}.png"
    )
    fig_path_f1 = plots_dir / (
        f"fig_B_valf1_aug={cfg.aug}_data={cfg.data_frac}_ep={cfg.epochs}_"
        f"cap={'full' if cfg.train_backbone else 'frozen'}_seed={cfg.seed}.png"
    )

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(df_curve["epoch"], df_curve["train_loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Model B Convergence (Train Loss)")
    plt.tight_layout()
    plt.savefig(fig_path_loss, dpi=200)
    plt.close()

    plt.figure()
    plt.plot(df_curve["epoch"], df_curve["val_f1"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation F1")
    plt.title("Model B Convergence (Validation F1)")
    plt.tight_layout()
    plt.savefig(fig_path_f1, dpi=200)
    plt.close()

    # final test metrics (after restoring best checkpoint)
    yt, pt = _eval(model, dl_te, device)
    metrics = _metrics_binary(yt, pt)

    # attach artifacts paths
    metrics["curve_path"] = str(curve_path)
    metrics["curve_fig_path_loss"] = str(fig_path_loss)
    metrics["curve_fig_path_f1"] = str(fig_path_f1)

    return metrics

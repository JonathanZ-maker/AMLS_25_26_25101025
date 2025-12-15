"""
Model A (SVM) - Training and evaluation.

Step 1:
- Train a single linear SVM with fixed hyperparameters (no grid search yet).
- Evaluate on test set with Acc/Prec/Rec/F1.
Step 2:
- Grid search over (C, gamma) using validation set
- Confusion matrix for interpretation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)


@dataclass(frozen=True)
class SVMGrid:
    """
    Hyperparameter grid.

    For RBF kernel:
        sweep over (C, gamma)
    For Linear kernel:
        sweep over C only (gamma ignored)
    """
    C_list: Tuple[float, ...] = (0.1, 1.0, 10.0)
    gamma_list: Tuple[float, ...] = (0.01, 0.1, 1.0)


@dataclass(frozen=True)
class SVMTrainConfig:
    """
    SVM training configuration.
    """
    kernel: str = "rbf"  # "rbf" or "linear"
    class_weight: Optional[str] = "balanced"


def train_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    C: float,
    gamma: Optional[float] = None,
    cfg: Optional[SVMTrainConfig] = None,
) -> SVC:
    """
    Train an SVM classifier.

    Args:
        x_train: (N, D) training features
        y_train: (N,) labels in {0, 1}
        C: regularization parameter
        gamma: RBF kernel coefficient (ignored for linear)
        cfg: SVMTrainConfig

    Returns:
        Trained sklearn.svm.SVC model
    """
    if cfg is None:
        cfg = SVMTrainConfig()

    if cfg.kernel == "linear":
        model = SVC(
            C=C,
            kernel="linear",
            class_weight=cfg.class_weight,
        )
    elif cfg.kernel == "rbf":
        if gamma is None:
            raise ValueError("gamma must be provided for RBF kernel.")
        model = SVC(
            C=C,
            gamma=gamma,
            kernel="rbf",
            class_weight=cfg.class_weight,
        )
    else:
        raise ValueError(f"Unsupported kernel: {cfg.kernel}")

    model.fit(x_train, y_train)
    return model


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Return confusion matrix counts TN, FP, FN, TP (binary {0,1}).
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def eval_binary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute metrics.

    Required / commonly reported:
      - accuracy, precision, recall, f1 (positive-class)
    Robust diagnostics:
      - f1_macro, bal_acc, pred_pos_rate
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "pred_pos_rate": float(np.mean(y_pred == 1)),
    }


def grid_search_svm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    grid: Optional[SVMGrid] = None,
    cfg: Optional[SVMTrainConfig] = None,
    selection_metric: str = "f1_macro",
) -> Tuple[float, Optional[float], Dict[str, float], List[Dict[str, float]]]:
    """
    Hyperparameter selection using validation set.

    For linear kernel:
        choose best C (gamma returned as None)
    For RBF kernel:
        choose best (C, gamma)

    Returns:
        best_C
        best_gamma (None for linear kernel)
        best_val_metrics
        trials: all attempted combinations and their val metrics
    """
    if grid is None:
        grid = SVMGrid()
    if cfg is None:
        cfg = SVMTrainConfig()

    best_score = -1.0
    best_C: Optional[float] = None
    best_gamma: Optional[float] = None
    best_metrics: Optional[Dict[str, float]] = None
    trials: List[Dict[str, float]] = []

    if cfg.kernel == "linear":
        for C in grid.C_list:
            model = train_svm(x_train, y_train, C=C, gamma=None, cfg=cfg)
            pred_val = model.predict(x_val)
            m = eval_binary(y_val, pred_val)

            trial = {"C": float(C), "gamma": np.nan, "kernel": "linear", **m}
            trials.append(trial)

            if selection_metric not in m:
                raise KeyError(f"selection_metric='{selection_metric}' not found in metrics.")
            score = float(m[selection_metric])
            if score > best_score:
                best_score = score
                best_C = float(C)
                best_gamma = None
                best_metrics = m

    elif cfg.kernel == "rbf":
        for C in grid.C_list:
            for gamma in grid.gamma_list:
                model = train_svm(x_train, y_train, C=C, gamma=gamma, cfg=cfg)
                pred_val = model.predict(x_val)
                m = eval_binary(y_val, pred_val)

                trial = {"C": float(C), "gamma": float(gamma), "kernel": "rbf", **m}
                trials.append(trial)

                if selection_metric not in m:
                    raise KeyError(f"selection_metric='{selection_metric}' not found in metrics.")
                score = float(m[selection_metric])
                if score > best_score:
                    best_score = score
                    best_C = float(C)
                    best_gamma = float(gamma)
                    best_metrics = m
    else:
        raise ValueError(f"Unsupported kernel: {cfg.kernel}")

    assert best_C is not None and best_metrics is not None
    return best_C, best_gamma, best_metrics, trials


def train_and_eval(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    C: float,
    gamma: Optional[float],
    cfg: SVMTrainConfig,
) -> Dict[str, float]:
    """
    Train once and evaluate on test set. Returns metrics + confusion counts.
    """
    model = train_svm(x_train, y_train, C=C, gamma=gamma, cfg=cfg)
    pred = model.predict(x_test)
    m = eval_binary(y_test, pred)
    cm = confusion_counts(y_test, pred)
    return {**m, **cm}
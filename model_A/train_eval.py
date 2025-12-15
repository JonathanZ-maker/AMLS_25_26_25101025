"""
Model A (SVM) - Training and evaluation.

Step 1:
- Train a single RBF SVM with fixed hyperparameters (no grid search yet).
- Evaluate on test set with Acc/Prec/Rec/F1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@dataclass(frozen=True)
class SVMBaselineConfig:
    """Configuration for Step 1 baseline SVM."""
    C: float = 1.0
    gamma: float = 0.1
    kernel: str = "rbf"


def train_svm_baseline(x_train: np.ndarray, y_train: np.ndarray, cfg: SVMBaselineConfig) -> SVC:
    """
    Train an RBF-kernel SVM classifier.

    Args:
        x_train: (N, D)
        y_train: (N,)
        cfg: SVM hyperparameters

    Returns:
        Trained sklearn SVC model
    """
    model = SVC(
        C=cfg.C,
        gamma=cfg.gamma,
        kernel=cfg.kernel,
        probability=False
    )
    model.fit(x_train, y_train)
    return model


def eval_binary_classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute standard binary classification metrics.

    Returns:
        dict with accuracy, precision, recall, f1
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

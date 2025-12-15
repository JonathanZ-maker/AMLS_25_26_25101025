"""
Model A (SVM) - Data loading for BreastMNIST.

This module supports two loading routes:
1) Local file under Datasets/BreastMNIST (e.g., .npz) for assessment usage.
2) medmnist package download for development usage.

Do NOT save any processed dataset to disk (per assignment rules).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class BreastMNISTSplits:
    """Container for dataset splits."""
    x_train: np.ndarray  # shape: (N, 28, 28) or (N, 28, 28, 1)
    y_train: np.ndarray  # shape: (N,)
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _standardize_shapes(x: np.ndarray) -> np.ndarray:
    """
    Ensure image array shape is (N, 28, 28).
    Accepts (N, 28, 28) or (N, 28, 28, 1).
    """
    if x.ndim == 4 and x.shape[-1] == 1:
        return x[..., 0]
    if x.ndim == 3:
        return x
    raise ValueError(f"Unexpected image shape: {x.shape}")


def _standardize_labels(y: np.ndarray) -> np.ndarray:
    """
    Ensure labels are 1D int array (N,).
    Accepts (N, 1) or (N,).
    """
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    return y.astype(np.int64)


def load_from_local_npz(dataset_dir: Path) -> Optional[BreastMNISTSplits]:
    """
    Try loading BreastMNIST splits from a local .npz file.

    Expected location:
        Datasets/BreastMNIST/*.npz

    Expected keys (common conventions):
        x_train, y_train, x_val, y_val, x_test, y_test

    Returns:
        BreastMNISTSplits if found and loaded; otherwise None.
    """
    if not dataset_dir.exists():
        return None

    npz_files = sorted(dataset_dir.glob("*.npz"))
    if not npz_files:
        return None

    # Load the first .npz found
    npz_path = npz_files[0]
    data = np.load(npz_path)

    required_keys = {"x_train", "y_train", "x_val", "y_val", "x_test", "y_test"}
    if not required_keys.issubset(set(data.files)):
        # Local file exists but doesn't match expected keys; fall back to medmnist
        return None

    x_train = _standardize_shapes(data["x_train"])
    y_train = _standardize_labels(data["y_train"])
    x_val = _standardize_shapes(data["x_val"])
    y_val = _standardize_labels(data["y_val"])
    x_test = _standardize_shapes(data["x_test"])
    y_test = _standardize_labels(data["y_test"])

    return BreastMNISTSplits(x_train, y_train, x_val, y_val, x_test, y_test)


def load_from_medmnist() -> BreastMNISTSplits:
    """
    Load BreastMNIST splits via medmnist package (downloads if needed).

    Note:
        medmnist typically returns images in uint8 and labels in shape (N, 1).
    """
    try:
        from medmnist import BreastMNIST
    except ImportError as e:
        raise ImportError(
            "medmnist is not installed. Install via: pip install medmnist"
        ) from e

    # Download and prepare splits
    train_ds = BreastMNIST(split="train", download=True)
    val_ds = BreastMNIST(split="val", download=True)
    test_ds = BreastMNIST(split="test", download=True)

    x_train = _standardize_shapes(np.asarray(train_ds.imgs))
    y_train = _standardize_labels(np.asarray(train_ds.labels))
    x_val = _standardize_shapes(np.asarray(val_ds.imgs))
    y_val = _standardize_labels(np.asarray(val_ds.labels))
    x_test = _standardize_shapes(np.asarray(test_ds.imgs))
    y_test = _standardize_labels(np.asarray(test_ds.labels))

    return BreastMNISTSplits(x_train, y_train, x_val, y_val, x_test, y_test)


def load_breastmnist(root_dir: Path) -> BreastMNISTSplits:
    """
    Load BreastMNIST splits.

    Priority:
        1) Datasets/BreastMNIST/*.npz
        2) medmnist download route
    """
    local_dir = root_dir / "Datasets" / "BreastMNIST"
    local = load_from_local_npz(local_dir)
    if local is not None:
        return local
    return load_from_medmnist()

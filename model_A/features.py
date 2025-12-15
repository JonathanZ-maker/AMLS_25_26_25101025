"""
Model A (SVM) - Feature extraction.

Step 1 implements only:
- Raw flatten features (784-d)
- Basic normalization / standardization

HOG features will be added in Step 2.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def flatten_images(x: np.ndarray) -> np.ndarray:
    """
    Flatten images into vectors.

    Args:
        x: (N, 28, 28) array

    Returns:
        (N, 784) float32 array
    """
    x = x.astype(np.float32)
    return x.reshape(x.shape[0], -1)


def fit_standardizer(x_train: np.ndarray) -> StandardScaler:
    """
    Fit a StandardScaler on training features only.

    Returns:
        fitted scaler
    """
    scaler = StandardScaler()
    scaler.fit(x_train)
    return scaler


def apply_standardizer(scaler: StandardScaler, x: np.ndarray) -> np.ndarray:
    """
    Apply an existing StandardScaler.
    """
    return scaler.transform(x)

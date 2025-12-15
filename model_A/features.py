"""
Model A (SVM) - Feature extraction.

Step 1 implements only:
- Raw flatten features (784-d)
- Basic normalization / standardization
Step 2 adds:
- HOG features

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

def hog_features(x: np.ndarray,
                 pixels_per_cell=(4, 4),
                 cells_per_block=(2, 2),
                 orientations: int = 9) -> np.ndarray:
    """
    Extract HOG features for each image.

    Args:
        x: (N, 28, 28) float32 in [0, 1]

    Returns:
        (N, D) float32 features
    """
    try:
        from skimage.feature import hog
    except ImportError as e:
        raise ImportError("scikit-image is required for HOG. Install: pip install scikit-image") from e

    feats = []
    for i in range(x.shape[0]):
        f = hog(
            x[i],
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
            feature_vector=True
        )
        feats.append(f.astype(np.float32))
    return np.stack(feats, axis=0)


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

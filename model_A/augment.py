"""
Model A - Image-domain augmentations.

Augmentations are applied on images BEFORE feature extraction.
This is aligned with the report narrative and avoids "feature-space" augmentation.
"""

from __future__ import annotations
import numpy as np


def random_rotate_shift(images: np.ndarray, rng: np.random.Generator,
                        max_degrees: float = 15.0,
                        max_shift_frac: float = 0.10) -> np.ndarray:
    """
    Apply random small rotation and translation to each image.

    Notes:
    - Implemented with a lightweight approach using scipy.ndimage if available.
    - If scipy is not available, raises an informative error.

    Args:
        images: (N, 28, 28) float32 in [0, 1]
        rng: numpy random generator
        max_degrees: rotation range [-max_degrees, +max_degrees]
        max_shift_frac: max shift as fraction of image size

    Returns:
        augmented images with same shape
    """
    try:
        from scipy.ndimage import rotate, shift
    except ImportError as e:
        raise ImportError("scipy is required for rotate/shift augmentation.") from e

    n, h, w = images.shape
    out = np.empty_like(images)

    max_shift_px = int(round(max_shift_frac * h))
    for i in range(n):
        angle = rng.uniform(-max_degrees, max_degrees)
        sx = rng.integers(-max_shift_px, max_shift_px + 1)
        sy = rng.integers(-max_shift_px, max_shift_px + 1)

        img = images[i]
        img = rotate(img, angle=angle, reshape=False, order=1, mode="nearest")
        img = shift(img, shift=(sx, sy), order=1, mode="nearest")
        out[i] = np.clip(img, 0.0, 1.0)

    return out


def add_gaussian_noise(images: np.ndarray, rng: np.random.Generator, sigma: float = 0.05) -> np.ndarray:
    """
    Add Gaussian noise and clip back to [0, 1].

    Args:
        images: (N, 28, 28) float32 in [0, 1]
        sigma: std of noise

    Returns:
        noisy images
    """
    noise = rng.normal(0.0, sigma, size=images.shape).astype(np.float32)
    return np.clip(images + noise, 0.0, 1.0)

"""
Model B - Data pipeline (PyTorch) for BreastMNIST.

- Loads splits via model_A.data.load_breastmnist (local npz preferred, else medmnist).
- Applies transforms (augmentation) on TRAIN only.
- Supports training-data subsampling for budget experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model_A.data import load_breastmnist  # reuse your proven loader


class NumpyBreastDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, transform=None):
        self.x = x  # (N, 28, 28) float32 in [0,1]
        self.y = y.astype(np.int64)  # (N,)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        img = self.x[idx]  # (28,28)
        label = int(self.y[idx])

        # to torch: (1,28,28)
        t = torch.from_numpy(img).float().unsqueeze(0)

        if self.transform is not None:
            t = self.transform(t)

        return t, label


def _subsample_indices(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if frac >= 0.999:
        return np.arange(len(y))

    rng = np.random.default_rng(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    n0 = max(1, int(len(idx0) * frac))
    n1 = max(1, int(len(idx1) * frac))
    sel0 = rng.choice(idx0, size=n0, replace=False)
    sel1 = rng.choice(idx1, size=n1, replace=False)
    sel = np.concatenate([sel0, sel1])
    rng.shuffle(sel)
    return sel


def get_splits_as_numpy(root_dir) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    splits = load_breastmnist(root_dir)
    x_train = splits.x_train.astype(np.float32) / 255.0
    y_train = splits.y_train.astype(np.int64)
    x_val = splits.x_val.astype(np.float32) / 255.0
    y_val = splits.y_val.astype(np.int64)
    x_test = splits.x_test.astype(np.float32) / 255.0
    y_test = splits.y_test.astype(np.int64)
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_transform(aug: str):
    """
    aug: "NoAug" | "RotShift" | "RotShift+Noise"
    Works on torch tensor (1,28,28). Keep transforms simple & deterministic per-call.
    """
    import torch
    import torch.nn.functional as F

    def noaug(x):
        return x

    def rotshift(x):
        # small random affine: rotate +/-15 deg, translate +/-2 px
        # implemented via affine_grid + grid_sample
        angle = (torch.rand(1).item() * 30.0 - 15.0) * np.pi / 180.0
        tx = (torch.rand(1).item() * 4.0 - 2.0) / 14.0  # normalize to [-1,1] approx
        ty = (torch.rand(1).item() * 4.0 - 2.0) / 14.0

        c, s = np.cos(angle), np.sin(angle)
        theta = torch.tensor([[c, -s, tx],
                              [s,  c, ty]], dtype=torch.float32).unsqueeze(0)  # (1,2,3)

        grid = F.affine_grid(theta, size=(1, 1, 28, 28), align_corners=False)
        x2 = x.unsqueeze(0)
        out = F.grid_sample(x2, grid, align_corners=False, padding_mode="zeros")
        return out.squeeze(0)

    def noise(x):
        return torch.clamp(x + 0.05 * torch.randn_like(x), 0.0, 1.0)

    if aug == "NoAug":
        return noaug
    if aug == "RotShift":
        return rotshift
    if aug == "RotShift+Noise":
        return lambda x: noise(rotshift(x))
    raise ValueError(f"Unknown augmentation: {aug}")


def get_dataloaders(root_dir, aug: str, data_frac: float, batch_size: int, seed: int, num_workers: int = 0):
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = get_splits_as_numpy(root_dir)

    sel = _subsample_indices(y_tr, data_frac, seed)
    x_tr = x_tr[sel]
    y_tr = y_tr[sel]

    tr_tf = build_transform(aug)
    va_tf = build_transform("NoAug")
    te_tf = build_transform("NoAug")

    ds_tr = NumpyBreastDataset(x_tr, y_tr, transform=tr_tf)
    ds_va = NumpyBreastDataset(x_va, y_va, transform=va_tf)
    ds_te = NumpyBreastDataset(x_te, y_te, transform=te_tf)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dl_tr, dl_va, dl_te

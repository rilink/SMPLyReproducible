"""
Shared in-memory-array dataset/scoring helpers, used wherever real and
synthetic windows need to be freely subset, combined, and evaluated together
(placement_validation.py, orientation_sweep.py, augmented_training.py).
Split out from placement_validation.py since both experiment folders depend
on it, not just Experiment 1.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

EXPERIMENTS_DIR = Path(__file__).resolve().parent
MODELS_DIR = EXPERIMENTS_DIR.parent / "models"
sys.path.insert(0, str(MODELS_DIR))

import common  # noqa: E402


class ArrayWindowDataset(Dataset):
    """Same (x, y, idx) interface as common.WindowDataset, but built from
    in-memory arrays so real and synthetic windows can be freely combined.
    """

    def __init__(self, X: np.ndarray, labels: np.ndarray):
        self.X = X.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.tensor(self.X[idx]), torch.tensor(self.labels[idx]), idx


def _labels_from_activity(activity_arr: np.ndarray) -> np.ndarray:
    activity_to_label = {a: i for i, a in enumerate(common.ACTIVITIES)}
    return np.array([activity_to_label[a] for a in activity_arr], dtype=np.int64)


def _subset(data: dict, subjects: list[str]):
    mask = np.isin(data["subject"], subjects)
    return data["X"][mask], _labels_from_activity(data["activity"][mask])


def _concat(pair_a, pair_b):
    return np.concatenate([pair_a[0], pair_b[0]], axis=0), np.concatenate([pair_a[1], pair_b[1]], axis=0)


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int = 64) -> DataLoader:
    return DataLoader(ArrayWindowDataset(X, y), batch_size=batch_size, shuffle=False)


def _score(yt: np.ndarray, yp: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
    }

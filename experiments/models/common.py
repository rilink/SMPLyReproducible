"""Shared training/eval utilities for the deep HAR models: fold loading,
checkpointed training, evaluation, and 5-fold cross-validation."""

import json
import random
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

PREPROCESSING_DIR = Path(__file__).resolve().parent.parent / "preprocessing"
sys.path.insert(0, str(PREPROCESSING_DIR))
import config as preprocessing_config  # noqa: E402

PROCESSED_DIR = PREPROCESSING_DIR / "processed"
WINDOWS_PATH = PROCESSED_DIR / "windows.npz"
FOLDS_PATH = PROCESSED_DIR / "folds.json"

CHECKPOINTS_DIR = Path(__file__).resolve().parent / "checkpoints"

ACTIVITIES = preprocessing_config.ACTIVITIES  # fixed label order, index = class id
NUM_CLASSES = len(ACTIVITIES)
NUM_CHANNELS = 6  # acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z
WINDOW_SIZE = preprocessing_config.WINDOW_SAMPLES


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def instance_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    return (x - mean) / (std + eps)


def load_folds() -> list[dict]:
    return json.loads(FOLDS_PATH.read_text())


class WindowDataset(Dataset):
    """Wraps a position-filtered slice of windows.npz for a given set of subjects."""

    def __init__(self, position: str, subjects: list[str]):
        npz = np.load(WINDOWS_PATH, allow_pickle=True)
        position_mask = npz["position"] == position
        subject_mask = np.isin(npz["subject"], subjects)
        mask = position_mask & subject_mask
        if not mask.any():
            raise ValueError(f"No windows found for position={position!r}, subjects={subjects}")

        self.X = npz["X"][mask].astype(np.float32)
        activity_to_label = {a: i for i, a in enumerate(ACTIVITIES)}
        self.labels = np.array([activity_to_label[a] for a in npz["activity"][mask]], dtype=np.int64)
        self.subjects = npz["subject"][mask]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.tensor(self.X[idx]), torch.tensor(self.labels[idx]), idx


def pooled_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
        "labels": labels,
    }


def train_one_fold(
    train_ds: WindowDataset,
    device: torch.device,
    make_model: Callable[[], nn.Module],
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 3e-4,
    val_fraction: float = 0.1,
    seed: int = 42,
    checkpoint_path: Path | None = None,
):
    """Train with an internal train/val split, AdamW + cosine LR + class-weighted CE.
    Keeps the model state from the epoch with the best validation macro-F1.

    If `checkpoint_path` is given and already exists, loads it and skips training
    entirely. Otherwise trains as usual and saves the best state to
    `checkpoint_path` (if given) before returning -- training on CPU is expensive
    enough here that a trained model should never be silently thrown away.
    """
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        print(f"    loading cached checkpoint: {checkpoint_path}")
        model = make_model().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return model

    set_seed(seed)
    n_val = int(len(train_ds) * val_fraction)
    n_train = len(train_ds) - n_val
    generator = torch.Generator().manual_seed(seed)
    tr_set, val_set = random_split(train_ds, [n_train, n_val], generator=generator)
    tr_loader = DataLoader(tr_set, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = make_model().to(device)

    counts = np.bincount(train_ds.labels, minlength=NUM_CLASSES).astype(float)
    class_weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32)
    class_weights = (class_weights / class_weights.sum() * NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_f1, best_state, best_epoch = 0.0, None, 0

    for epoch in range(epochs):
        model.train()
        for x, y, _ in tr_loader:
            x, y = instance_norm(x.to(device)), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        val_yt, val_yp = [], []
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = instance_norm(x.to(device))
                val_yp.extend(model(x).argmax(1).cpu().numpy())
                val_yt.extend(y.numpy())
        val_f1 = f1_score(val_yt, val_yp, average="macro", zero_division=0)
        val_acc = accuracy_score(val_yt, val_yp)

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        print(f"    epoch {epoch+1:02d}/{epochs}  val_acc={val_acc:.3f}  val_f1={val_f1:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"    best epoch={best_epoch}  best_val_f1={best_f1:.4f}")

    if checkpoint_path is not None:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"    saved checkpoint: {checkpoint_path}")

    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    for x, y, _ in loader:
        x = instance_norm(x.to(device))
        y_pred.extend(model(x).argmax(1).cpu().numpy())
        y_true.extend(y.numpy())
    return np.array(y_true), np.array(y_pred)


def run_cv(
    make_model: Callable[[], nn.Module],
    model_tag: str,
    position: str = "forearm",
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 3e-4,
    seed: int = 42,
    device: torch.device | None = None,
):
    """Run the project's 5-fold leave-3-subjects-out CV for one model architecture
    on windows from a single body position. Prints and returns per-fold and pooled
    (accuracy, macro-F1) metrics.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{model_tag}] position={position}  device={device}")

    folds = load_folds()
    fold_metrics = {}
    all_yt, all_yp = [], []

    for fold in folds:
        fold_id = fold["fold"]
        print(f"\n=== fold {fold_id}: test subjects {fold['test_subjects']} ===")

        train_ds = WindowDataset(position, fold["train_subjects"])
        test_ds = WindowDataset(position, fold["test_subjects"])

        checkpoint_path = CHECKPOINTS_DIR / f"{model_tag}_{position}_fold{fold_id}.pt"
        model = train_one_fold(
            train_ds, device, make_model, epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
            checkpoint_path=checkpoint_path,
        )

        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        yt, yp = evaluate(model, test_loader, device)
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
        print(f"  test  acc={acc:.3f}  macro_f1={f1:.3f}")

        fold_metrics[fold_id] = {"accuracy": float(acc), "macro_f1": float(f1)}
        all_yt.extend(yt.tolist())
        all_yp.extend(yp.tolist())

    accs = np.array([v["accuracy"] for v in fold_metrics.values()])
    f1s = np.array([v["macro_f1"] for v in fold_metrics.values()])
    pooled = pooled_metrics(all_yt, all_yp)

    print(f"\n====== {model_tag} CV SUMMARY (position={position}) ======")
    print(f"Per-fold accuracy:  {accs.mean():.3f} +/- {accs.std():.3f}")
    print(f"Per-fold macro-F1:  {f1s.mean():.3f} +/- {f1s.std():.3f}")
    print(f"Pooled accuracy={pooled['accuracy']:.3f}  pooled macro_f1={pooled['macro_f1']:.3f}")

    return {"fold_metrics": fold_metrics, "pooled": pooled}

"""
Sweep the in-plane (Z) orientation at a fixed vertex -- default vertex 1922,
the same forearm location used for placement B (the plane the user has been
rotating interactively) -- in fixed-degree increments, and evaluate the
already-trained model_A checkpoints on each resulting synthetic placement.

No retraining happens: model_A depends only on real placement A, which never
changes across this sweep, so every angle is pure evaluation against the
checkpoints saved by baseline.py.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = EXPERIMENTS_DIR.parent / "models"
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR))

import common  # noqa: E402
from synthesize_placements import (  # noqa: E402
    load_A_context,
    rotation_A_to_vertex,
    build_synthetic_dataset_for_rotation,
)
from dataset_utils import (  # noqa: E402
    ArrayWindowDataset,
    _loader,
    _score,
    _subset,
)


def sweep_orientation(
    make_model,
    model_tag: str,
    vertex_idx: int = 1922,
    step_deg: int = 10,
    z_degs: list[int] | None = None,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 3e-4,
    seed: int = 42,
    device: torch.device | None = None,
    folds: list[dict] | None = None,
) -> list[dict]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = folds if folds is not None else common.load_folds()
    z_degs = z_degs if z_degs is not None else list(range(0, 360, step_deg))
    print(f"[orientation-sweep:{model_tag}] vertex={vertex_idx}  z_degs={z_degs}  device={device}")

    real_npz = np.load(common.WINDOWS_PATH, allow_pickle=True)
    real_mask = real_npz["position"] == "forearm"
    real = {k: real_npz[k][real_mask] for k in ["X", "activity", "subject"]}

    # Load (or train once, if truly missing) the model_A checkpoint per fold.
    # These depend only on real A, so they are identical to -- and reused
    # directly from -- the checkpoints saved by baseline.py (Experiment 0).
    models_by_fold = {}
    for fold in folds:
        fold_id = fold["fold"]
        X_A_train, y_A_train = _subset(real, fold["train_subjects"])
        train_ds_a = ArrayWindowDataset(X_A_train, y_A_train)
        checkpoint_a = common.CHECKPOINTS_DIR / f"{model_tag}_forearm_fold{fold_id}.pt"
        models_by_fold[fold_id] = common.train_one_fold(
            train_ds_a, device, make_model, epochs, batch_size, lr, seed=seed,
            checkpoint_path=checkpoint_a,
        )

    context = load_A_context()
    results = []

    for z_deg in z_degs:
        R, angle_deg = rotation_A_to_vertex(vertex_idx, z_deg, context=context)
        synth = build_synthetic_dataset_for_rotation(R, position="forearm")

        fold_scores = []
        for fold in folds:
            X_test, y_test = _subset(synth, fold["test_subjects"])
            yt, yp = common.evaluate(models_by_fold[fold["fold"]], _loader(X_test, y_test), device)
            fold_scores.append(_score(yt, yp))

        accs = np.array([s["accuracy"] for s in fold_scores])
        f1s = np.array([s["macro_f1"] for s in fold_scores])
        row = {
            "z_deg": z_deg,
            "angular_distance_deg": angle_deg,
            "mean_accuracy": float(accs.mean()),
            "std_accuracy": float(accs.std()),
            "mean_macro_f1": float(f1s.mean()),
            "std_macro_f1": float(f1s.std()),
            "fold_scores": fold_scores,
        }
        results.append(row)
        print(
            f"z={z_deg:3d}deg  angle_from_A={angle_deg:6.2f}deg  "
            f"acc={row['mean_accuracy']:.3f}+/-{row['std_accuracy']:.3f}  "
            f"f1={row['mean_macro_f1']:.3f}+/-{row['std_macro_f1']:.3f}"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="deepconvlstm",
        choices=["deepconvlstm", "cnnhar", "tinyhar"],
    )
    parser.add_argument("--vertex", type=int, default=1922)
    parser.add_argument("--step-deg", type=int, default=10)
    parser.add_argument("--z-degs", type=int, nargs="+", default=None,
                         help="Explicit list of z-degree twists to evaluate (overrides --step-deg)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_module = importlib.import_module(f"model_{args.model}")
    results = sweep_orientation(
        model_module.make_model, model_tag=args.model, vertex_idx=args.vertex,
        step_deg=args.step_deg, z_degs=args.z_degs, epochs=args.epochs, batch_size=args.batch_size,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved sweep results -> {args.output}")

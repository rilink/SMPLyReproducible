"""
Experiment 2: adds rotation-jittered copies on top of the real training data,
rather than replacing it, for every fold:

  - all 12 real, unrotated train-subject recordings, PLUS
  - 4 groups of 3 subjects, each twisted +inner, -inner, +outer, or -outer
    deg from A (--inner-deg/--outer-deg, default 30/60),

all about A's own vertex (2208) surface normal, the same family used
throughout the orientation sweep. Effective training set doubles to 24
subject-copies. Each subject is assigned to exactly one rotation group
(seeded, reproducible 3/3/3/3 split) plus their own unrotated recording.

Evaluated at the same offsets used in training -- A (0) and +-inner/+-outer
deg, not the full sweep grid, since the model never trained on wider angles.
The original Experiment-0 checkpoint is evaluated at the same offsets, with
no retraining, for a direct before/after comparison.
"""

import argparse
import importlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = EXPERIMENTS_DIR.parent / "models"
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR))

import common  # noqa: E402
from synthesize_placements import load_A_context, rotation_A_to_vertex, build_synthetic_dataset_for_rotation  # noqa: E402
from dataset_utils import ArrayWindowDataset, _subset, _concat, _score, _loader  # noqa: E402

VERTEX_A = 2208  # placement A's own vertex, every rotation here twists about its normal
VERTEX_A_Z_DEG = 300  # placement A's own twist at that vertex (forearm_placements.yaml)
CHECKPOINTS_DIR = EXPERIMENTS_DIR / "checkpoints"


def rotation_groups_deg(inner_deg: int, outer_deg: int) -> dict[str, int]:
    """+/-inner and +/-outer offsets from A's own twist. "left" = minus,
    "right" = plus -- flip the sign convention here if needed, nothing else
    has to change."""
    return {f"+{inner_deg}": inner_deg, f"-{inner_deg}": -inner_deg,
            f"+{outer_deg}": outer_deg, f"-{outer_deg}": -outer_deg}


def eval_offsets_deg(inner_deg: int, outer_deg: int) -> list[int]:
    """The same angles used in training, plus 0 (real A), not the full 0-360
    sweep grid -- see module docstring for why."""
    return [-outer_deg, -inner_deg, 0, inner_deg, outer_deg]


def assign_rotation_groups(train_subjects: list[str], seed: int, groups_deg: dict[str, int]) -> dict[str, list[str]]:
    """Deterministically split 12 train subjects into four groups of 3, one
    per rotation offset, so every subject contributes exactly one rotated
    copy in addition to their own unrotated recording."""
    rng = random.Random(seed)
    shuffled = rng.sample(sorted(train_subjects), len(train_subjects))
    keys = list(groups_deg.keys())
    return {keys[i]: shuffled[i * 3:(i + 1) * 3] for i in range(4)}


def build_augmented_training_set(real: dict, train_subjects: list[str], seed: int, context, groups_deg: dict[str, int]):
    """Real (unrotated) data for all train_subjects, plus one rotated copy per
    subject (offset determined by assign_rotation_groups), concatenated on top
    -- this doubles the effective training set, it never replaces anything."""
    groups = assign_rotation_groups(train_subjects, seed, groups_deg)

    X_all, y_all = _subset(real, train_subjects)  # all 12, unrotated, kept as-is
    for group_key, subjects in groups.items():
        offset_deg = groups_deg[group_key]
        z_deg = (VERTEX_A_Z_DEG + offset_deg) % 360
        R, _ = rotation_A_to_vertex(VERTEX_A, z_deg, context=context)
        X_sub, y_sub = _subset(real, subjects)
        X_rot = X_sub.copy()
        X_rot[..., 0:3] = X_sub[..., 0:3] @ R.T
        X_rot[..., 3:6] = X_sub[..., 3:6] @ R.T
        X_all, y_all = _concat((X_all, y_all), (X_rot, y_sub))

    return X_all, y_all, groups


def _score_at_offsets(model, real, synth_context, test_subj: list[str], device, position: str, offsets_deg: list[int]) -> dict:
    """Evaluate one already-trained model at every offset in offsets_deg.
    Offset 0 uses real, unrotated test data directly (exact, no rotation
    applied at all), nonzero offsets use the same-vertex synthetic rotation."""
    scores = {}
    for offset_deg in offsets_deg:
        if offset_deg == 0:
            X_test, y_test = _subset(real, test_subj)
            angle_deg = 0.0
        else:
            z_deg = (VERTEX_A_Z_DEG + offset_deg) % 360
            R, angle_deg = rotation_A_to_vertex(VERTEX_A, z_deg, context=synth_context)
            synth = build_synthetic_dataset_for_rotation(R, position=position)
            X_test, y_test = _subset(synth, test_subj)
        yt, yp = common.evaluate(model, _loader(X_test, y_test), device)
        score = _score(yt, yp)
        score["angular_distance_deg"] = angle_deg
        scores[offset_deg] = score
    return scores


def run_augmented_training(
    make_model,
    model_tag: str,
    inner_deg: int = 30,
    outer_deg: int = 60,
    seed: int = 42,
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: torch.device | None = None,
    folds: list[dict] | None = None,
    position: str = "forearm",
    output_path: Path | None = None,
) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = folds if folds is not None else common.load_folds()
    context = load_A_context()
    groups_deg = rotation_groups_deg(inner_deg, outer_deg)
    offsets_deg = eval_offsets_deg(inner_deg, outer_deg)
    tag = f"{inner_deg}_{outer_deg}"

    real_npz = np.load(common.WINDOWS_PATH, allow_pickle=True)
    real_mask = real_npz["position"] == position
    real = {k: real_npz[k][real_mask] for k in ["X", "activity", "subject"]}

    results = {"augmented": {}, "baseline": {}}
    group_assignments = {}

    for fold in folds:
        fold_id = fold["fold"]
        train_subj, test_subj = fold["train_subjects"], fold["test_subjects"]
        print(f"\n=== fold {fold_id}: test={test_subj} ===")

        X_train, y_train, groups = build_augmented_training_set(real, train_subj, seed, context, groups_deg)
        group_assignments[fold_id] = groups
        print(f"  rotation groups: {groups}")
        print(f"  augmented training set: {len(X_train)} windows (vs. {len(_subset(real, train_subj)[0])} unaugmented)")

        train_ds = ArrayWindowDataset(X_train, y_train)
        aug_checkpoint = CHECKPOINTS_DIR / f"{model_tag}_augmented{tag}_seed{seed}_fold{fold_id}.pt"
        aug_model = common.train_one_fold(
            train_ds, device, make_model, epochs, batch_size, lr, seed=seed,
            checkpoint_path=aug_checkpoint,
        )
        results["augmented"][fold_id] = _score_at_offsets(aug_model, real, context, test_subj, device, position, offsets_deg)
        print(f"   augmented: {results['augmented'][fold_id]}")

        # Baseline: the original, non-augmented model from Experiment 0 --
        # already trained, so this only loads its checkpoint, no retraining
        # (train_one_fold returns early whenever checkpoint_path already
        # exists, without touching train_ds at all).
        base_train_ds = ArrayWindowDataset(*_subset(real, train_subj))
        base_checkpoint = common.CHECKPOINTS_DIR / f"{model_tag}_{position}_fold{fold_id}.pt"
        base_model = common.train_one_fold(
            base_train_ds, device, make_model, epochs, batch_size, lr, seed=seed,
            checkpoint_path=base_checkpoint,
        )
        results["baseline"][fold_id] = _score_at_offsets(base_model, real, context, test_subj, device, position, offsets_deg)
        print(f"   baseline:  {results['baseline'][fold_id]}")

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps({
                "results": results, "group_assignments": group_assignments,
                "folds_completed": sorted(results["augmented"].keys()),
            }, indent=2))
            print(f"   [saved partial results -> {output_path}]")

    return {"results": results, "group_assignments": group_assignments}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tinyhar", choices=["deepconvlstm", "cnnhar", "tinyhar"])
    parser.add_argument("--inner-deg", type=int, default=30, help="Smaller +/- rotation offset")
    parser.add_argument("--outer-deg", type=int, default=60, help="Larger +/- rotation offset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    model_module = importlib.import_module(f"model_{args.model}")
    output = run_augmented_training(
        model_module.make_model, model_tag=args.model, inner_deg=args.inner_deg, outer_deg=args.outer_deg,
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, output_path=args.output,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))
        print(f"\nSaved results -> {args.output}")

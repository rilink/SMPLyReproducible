"""
Experiment 1, LightGBM: same-vertex (vertex 2208, twist-only) orientation
sweep, analogous to orientation_sweep.py -- "model A" is fit once per fold on
real placement-A features and reused across every angle, mirroring the
PyTorch sweep's no-retraining design. Unlike the PyTorch case, each angle
still needs a fresh, expensive feature-extraction pass (~16 min for the full
forearm population), since LightGBM's features come from the raw rotated
signal rather than being learned end-to-end. Each angle's features are
cached under `models/processed/lightgbm_features_sweep_v{vertex}_z{deg}.csv`,
so re-running a single angle later is free. Results are written incrementally
after every angle.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = EXPERIMENTS_DIR.parent / "models"
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(EXPERIMENTS_DIR))

import common  # noqa: E402
import model_lightgbm  # noqa: E402
from synthesize_placements import (  # noqa: E402
    load_A_context,
    rotation_A_to_vertex,
    build_synthetic_dataset_for_rotation,
)


def sweep_orientation_lightgbm(
    z_degs: list[int],
    vertex_idx: int = 2208,
    position: str = "forearm",
    output_path: Path | None = None,
) -> list[dict]:
    print(f"[orientation-sweep-lightgbm] vertex={vertex_idx}  z_degs={z_degs}")

    real_df = model_lightgbm.get_or_build_features(position)
    folds = common.load_folds()

    # fit once per fold, reused across every angle -- see module docstring for why
    models_by_fold = {}
    for fold in folds:
        fold_id = fold["fold"]
        train_df = real_df[real_df["subject"].isin(fold["train_subjects"])]
        models_by_fold[fold_id] = model_lightgbm.fit_lgbm(train_df)
        print(f"  fitted model_A for fold {fold_id}")

    context = load_A_context()
    results = []

    for z_deg in z_degs:
        R, angle_deg = rotation_A_to_vertex(vertex_idx, z_deg, context=context)
        synth = build_synthetic_dataset_for_rotation(R, position=position)

        cache_name = f"sweep_v{vertex_idx}_z{z_deg}"
        synth_df = model_lightgbm.get_or_build_features_from_arrays(
            synth["X"], synth["activity"], synth["subject"], cache_name=cache_name,
        )

        fold_scores = []
        for fold in folds:
            fold_id = fold["fold"]
            test_subj = fold["test_subjects"]
            model, activity_to_label = models_by_fold[fold_id]
            yt, yp = model_lightgbm.predict_lgbm(
                model, activity_to_label, synth_df[synth_df["subject"].isin(test_subj)],
            )
            fold_scores.append({
                "accuracy": float(accuracy_score(yt, yp)),
                "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            })

        accs = np.array([s["accuracy"] for s in fold_scores])
        f1s = np.array([s["macro_f1"] for s in fold_scores])
        row = {
            "z_deg": z_deg,
            "angular_distance_deg": angle_deg,
            "mean_accuracy": float(accs.mean()), "std_accuracy": float(accs.std()),
            "mean_macro_f1": float(f1s.mean()), "std_macro_f1": float(f1s.std()),
            "fold_scores": fold_scores,
        }
        results.append(row)
        print(
            f"z={z_deg:3d}deg  angle_from_A={angle_deg:6.2f}deg  "
            f"acc={row['mean_accuracy']:.3f}+/-{row['std_accuracy']:.3f}  "
            f"f1={row['mean_macro_f1']:.3f}+/-{row['std_macro_f1']:.3f}"
        )

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2))
            print(f"   [saved partial sweep results -> {output_path}]")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vertex", type=int, default=2208)
    parser.add_argument("--z-degs", type=int, nargs="+", default=list(range(0, 360, 30)),
                         help="Explicit list of z-degree twists to evaluate (default: 12-point, 30deg-step sweep)")
    parser.add_argument("--position", default="forearm")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = sweep_orientation_lightgbm(
        z_degs=args.z_degs, vertex_idx=args.vertex, position=args.position,
        output_path=args.output,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(f"\nSaved sweep results -> {args.output}")

import argparse
import json
from pathlib import Path

import numpy as np

import config
from cv_split import make_folds
from dataset import build_dataset


def main():
    parser = argparse.ArgumentParser(description="Build windowed RealWorld acc+gyr dataset (forearm position) with CV folds")
    parser.add_argument("--data-root", type=Path, default=config.DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "processed")
    parser.add_argument("--subjects", nargs="+", default=config.SUBJECTS)
    parser.add_argument("--activities", nargs="+", default=config.ACTIVITIES)
    parser.add_argument("--window-seconds", type=float, default=config.WINDOW_SECONDS)
    parser.add_argument("--overlap", type=float, default=config.OVERLAP)
    parser.add_argument("--target-hz", type=int, default=config.TARGET_HZ)
    parser.add_argument("--n-folds", type=int, default=config.N_FOLDS)
    parser.add_argument("--subjects-per-fold", type=int, default=config.SUBJECTS_PER_FOLD)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root: {args.data_root}")
    data = build_dataset(
        data_root=args.data_root,
        subjects=args.subjects,
        activities=args.activities,
        window_seconds=args.window_seconds,
        overlap=args.overlap,
        target_hz=args.target_hz,
    )

    dataset_path = args.output_dir / "windows.npz"
    np.savez_compressed(
        dataset_path,
        X=data["X"],
        activity=data["activity"],
        subject=data["subject"],
        position=data["position"],
        start_time_s=data["start_time_s"],
    )
    print(f"Saved {data['X'].shape[0]} windows -> {dataset_path}")

    folds = make_folds(args.subjects, args.n_folds, args.subjects_per_fold, args.seed)
    folds_path = args.output_dir / "folds.json"
    folds_path.write_text(json.dumps(folds, indent=2))
    print(f"Saved {len(folds)} folds -> {folds_path}")


if __name__ == "__main__":
    main()

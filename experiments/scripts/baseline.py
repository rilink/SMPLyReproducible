"""
Experiment 0: Baseline.

Train and evaluate each HAR model at a single, fixed sensor placement (the
reference placement A, the real RealWorld 'forearm' recordings), using the
project's standard 5-fold leave-3-subjects-out cross-validation. This is not a
hypothesis test -- it establishes the upper-bound, same-placement performance
against which the placement-mismatch effects in Experiments 1 and 2 are
measured.

Runs LightGBM, DeepConvLSTM, and TinyHAR by default on accelerometer+gyroscope,
3-second (150-sample) windows, forearm position only. CNN-HAR is temporarily
excluded from the default run for time reasons -- it remains a valid
`--models` choice and can be run later with `--models cnnhar`. Every deep
model's per-fold checkpoint is saved/reused via common.train_one_fold
(models/checkpoints/), LightGBM has no per-fold checkpoint (retrains each run --
its cost is dominated by one-time feature extraction, cached separately per
position). Combined results (per-fold accuracy/macro-F1 for every model run)
are written to processed/experiment0_baseline_results.json.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

EXPERIMENTS_DIR = Path(__file__).resolve().parent
MODELS_DIR = EXPERIMENTS_DIR.parent / "models"
sys.path.insert(0, str(MODELS_DIR))

import common  # noqa: E402
import model_lightgbm  # noqa: E402

DEEP_MODELS = ("deepconvlstm", "cnnhar", "tinyhar")
ALL_MODELS = ("lightgbm",) + DEEP_MODELS
# cnnhar is temporarily excluded from the default run for time reasons -- still
# a valid --models choice, just not run unless explicitly requested.
DEFAULT_MODELS = tuple(m for m in ALL_MODELS if m != "cnnhar")


def run_baseline(
    models: tuple[str, ...] = DEFAULT_MODELS,
    position: str = "forearm",
    epochs: int = 15,
    batch_size: int = 256,
    lr: float = 3e-4,
) -> dict:
    results = {}

    if "lightgbm" in models:
        print(f"\n{'=' * 60}\n[baseline] lightgbm\n{'=' * 60}")
        results["lightgbm"] = model_lightgbm.run_cv(position=position)

    for name in DEEP_MODELS:
        if name not in models:
            continue
        print(f"\n{'=' * 60}\n[baseline] {name}\n{'=' * 60}")
        model_module = importlib.import_module(f"model_{name}")
        results[name] = common.run_cv(
            model_module.make_model, model_tag=name, position=position,
            epochs=epochs, batch_size=batch_size, lr=lr,
        )

    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    print("\n====== Experiment 0: Baseline summary ======")
    for name, r in results.items():
        fold_scores = r["fold_metrics"]
        accs = np.array([v["accuracy"] for v in fold_scores.values()])
        f1s = np.array([v["macro_f1"] for v in fold_scores.values()])
        print(f"{name:14s}  accuracy={accs.mean():.3f}+/-{accs.std():.3f}  "
              f"macro_f1={f1s.mean():.3f}+/-{f1s.std():.3f}  "
              f"(pooled acc={r['pooled']['accuracy']:.3f}, pooled f1={r['pooled']['macro_f1']:.3f})")


def _json_safe(results: dict) -> dict:
    """pooled_metrics() includes a confusion_matrix ndarray and numpy-typed
    labels -- convert everything to plain JSON-serializable types."""
    safe = {}
    for name, r in results.items():
        pooled = dict(r["pooled"])
        pooled["confusion_matrix"] = np.asarray(pooled["confusion_matrix"]).tolist()
        pooled["labels"] = [int(x) for x in pooled["labels"]]
        safe[name] = {"fold_metrics": r["fold_metrics"], "pooled": pooled}
    return safe


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(ALL_MODELS))
    parser.add_argument("--position", default="forearm",
                         choices=["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path,
                         default=EXPERIMENTS_DIR / "processed" / "experiment0_baseline_results.json")
    args = parser.parse_args()

    results = run_baseline(
        models=tuple(args.models), position=args.position,
        epochs=args.epochs, batch_size=args.batch_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(_json_safe(results), indent=2))
    print(f"\nSaved combined results -> {args.output}")

from pathlib import Path

import numpy as np

from loading import load_sensor_parts
from windowing import make_windows

POSITION = "forearm"  # the only position used anywhere in this project


def build_dataset(
    data_root: Path,
    subjects: list[str],
    activities: list[str],
    window_seconds: float,
    overlap: float,
    target_hz: int,
):
    """Build windowed (acc, gyr) samples at the forearm position for every
    (subject, activity) combination. Returns a dict of parallel numpy arrays:
    X (n, n_samples, 6), activity, subject, position, start_time_s.
    """
    X_list, activity_list, subject_list, start_list = [], [], [], []

    for subject in subjects:
        subject_dir = data_root / subject
        subject_window_count = 0
        for activity in activities:
            acc_parts = load_sensor_parts(subject_dir, activity, "acc", POSITION)
            gyr_parts = load_sensor_parts(subject_dir, activity, "gyr", POSITION)
            if not acc_parts or not gyr_parts:
                print(f"  [skip] no acc/gyr data: {subject}/{activity}/{POSITION}")
                continue

            windows, starts = make_windows(acc_parts, gyr_parts, window_seconds, overlap, target_hz)
            if len(windows) == 0:
                continue

            X_list.append(windows)
            activity_list.extend([activity] * len(windows))
            subject_list.extend([subject] * len(windows))
            start_list.append(starts)
            subject_window_count += len(windows)

        print(f"[done] {subject}: {subject_window_count} windows")

    X = np.concatenate(X_list, axis=0) if X_list else np.empty((0, 0, 6), dtype=np.float32)
    return {
        "X": X,
        "activity": np.array(activity_list),
        "subject": np.array(subject_list),
        "position": np.full(len(X), POSITION),
        "start_time_s": np.concatenate(start_list, axis=0) if start_list else np.empty((0,)),
    }

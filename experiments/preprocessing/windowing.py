"""Aligns acc+gyr parts onto a common time grid and slices them into
fixed-length windows."""

import numpy as np
import pandas as pd


def _resample_xyz(df: pd.DataFrame, query_times_s: np.ndarray) -> np.ndarray:
    t = df["time_ms"].to_numpy() / 1000.0
    query = np.clip(query_times_s, t[0], t[-1])
    return np.stack([np.interp(query, t, df[axis].to_numpy()) for axis in ("x", "y", "z")], axis=1)


def _overlapping_pairs(acc_parts: list[pd.DataFrame], gyr_parts: list[pd.DataFrame], window_seconds: float):
    """Pair an acc part with a gyr part iff they overlap in time by at least
    one window's worth. Pairing by time (rather than by list index) is robust
    to acc/gyr having a different number of parts, and to the ~15-20 minute
    gaps between parts making any accidental cross-session pairing impossible.
    """
    min_overlap_ms = window_seconds * 1000
    pairs = []
    for acc_df in acc_parts:
        a_start, a_end = acc_df["time_ms"].iloc[0], acc_df["time_ms"].iloc[-1]
        for gyr_df in gyr_parts:
            g_start, g_end = gyr_df["time_ms"].iloc[0], gyr_df["time_ms"].iloc[-1]
            lo, hi = max(a_start, g_start), min(a_end, g_end)
            if hi - lo >= min_overlap_ms:
                pairs.append((acc_df, gyr_df))
    return pairs


def make_windows(
    acc_parts: list[pd.DataFrame],
    gyr_parts: list[pd.DataFrame],
    window_seconds: float,
    overlap: float,
    target_hz: int,
):
    """Slide fixed-length windows over each time-overlapping (acc, gyr) part
    pair. Returns (windows, start_times_s) where windows has shape
    (n_windows, n_samples, 6) with channel order [acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z].
    Windows never span a part boundary.
    """
    n_samples = int(round(window_seconds * target_hz))
    step_seconds = window_seconds * (1 - overlap)

    windows = []
    start_times = []
    for acc_df, gyr_df in _overlapping_pairs(acc_parts, gyr_parts, window_seconds):
        common_start = max(acc_df["time_ms"].iloc[0], gyr_df["time_ms"].iloc[0]) / 1000.0
        common_end = min(acc_df["time_ms"].iloc[-1], gyr_df["time_ms"].iloc[-1]) / 1000.0

        t0 = common_start
        while t0 + window_seconds <= common_end:
            query = np.linspace(t0, t0 + window_seconds, n_samples, endpoint=False)
            acc_win = _resample_xyz(acc_df, query)
            gyr_win = _resample_xyz(gyr_df, query)
            windows.append(np.concatenate([acc_win, gyr_win], axis=1))
            start_times.append(t0)
            t0 += step_seconds

    if not windows:
        return np.empty((0, n_samples, 6), dtype=np.float32), np.empty((0,), dtype=np.float64)
    return np.stack(windows).astype(np.float32), np.array(start_times, dtype=np.float64)

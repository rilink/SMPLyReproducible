"""
LightGBM on handcrafted IMU features, extracted from acc+gyr windows.
Input: 6 channels (acc_x,y,z, gyr_x,y,z), 150-frame windows at 50 Hz.
Features: statistical + frequency-domain descriptors per channel, plus 3D
(accelerometer/gyroscope) descriptors: orientation/trajectory, posture, signal
magnitude, axis correlation, spectral entropy, Hjorth parameters, fractal dimension.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.fftpack import fft
from scipy.signal import find_peaks, welch
from scipy.stats import entropy, iqr, kurtosis, skew

import common

CHANNEL_NAMES = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]


# ============================================================
# Feature calculation (signal -> scalar descriptors)
# ============================================================

def signal_magnitude_vector(window: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(np.square(window), axis=1))


def spectral_entropy(signal: np.ndarray, sf: int = 50) -> float:
    _, psd = welch(signal, sf)
    psd_norm = psd / np.sum(psd)
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-12))


def hjorth_parameters(signal: np.ndarray) -> tuple[float, float]:
    d1 = np.diff(signal)
    d2 = np.diff(d1)
    var0, var1, var2 = np.var(signal), np.var(d1), np.var(d2)
    mobility = np.sqrt(var1 / var0) if var0 > 0 else 0
    complexity = (np.sqrt(var2 / var1) / mobility) if var1 > 0 and mobility > 0 else 0
    return mobility, complexity


def mean_crossing_rate(signal: np.ndarray) -> float:
    mean_val = np.mean(signal)
    crossings = np.where(np.diff(np.signbit(signal - mean_val)))[0]
    return len(crossings) / len(signal)


def differential_entropy(signal: np.ndarray) -> float:
    return 0.5 * np.log(2 * np.pi * np.e * np.var(signal))


def petrosian_fd(signal: np.ndarray) -> float:
    diff = np.diff(signal)
    sign_changes = np.sum(diff[1:] * diff[:-1] < 0)
    n = len(signal)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * sign_changes)))


def katz_fd(signal: np.ndarray) -> float:
    L = np.sum(np.abs(np.diff(signal)))
    d = np.max(np.abs(signal - signal[0]))
    n = len(signal)
    return np.log10(n) / (np.log10(d / (L + 1e-6) + 1e-6))


def orientation(x: np.ndarray, y: np.ndarray, z: np.ndarray, dt: float = 0.02):
    acc = np.stack([x, y, z], axis=1)
    vel = np.cumsum(acc * dt, axis=0)
    pos = np.cumsum(vel * dt, axis=0)
    pos_x, pos_y, pos_z = pos[-1]
    return pos_x, pos_y, pos_z, np.sum(vel[-1]), np.sum(pos[-1])


def calculate_1d_features(signal: np.ndarray, axis_name: str) -> dict:
    result = {}
    fft_vals = np.abs(fft(signal))[: len(signal) // 2]

    result[f"{axis_name}_mean"] = np.mean(signal)
    result[f"{axis_name}_var"] = np.var(signal)
    rms = np.sqrt(np.mean(signal**2))
    result[f"{axis_name}_rms"] = rms
    result[f"{axis_name}_shape_factor"] = rms / (abs(np.mean(signal)) + 1e-6)
    result[f"{axis_name}_min"] = np.min(signal)
    result[f"{axis_name}_max"] = np.max(signal)
    result[f"{axis_name}_median"] = np.median(signal)
    result[f"{axis_name}_iqr"] = iqr(signal)
    result[f"{axis_name}_sma"] = np.sum(np.abs(signal)) / len(signal)
    result[f"{axis_name}_skewness"] = skew(signal)
    result[f"{axis_name}_kurtosis"] = kurtosis(signal)

    j = np.diff(signal)
    result[f"{axis_name}_jerk_mean"] = np.mean(np.abs(j))
    result[f"{axis_name}_jerk_std"] = np.std(j)

    peaks, _ = find_peaks(signal, distance=3)
    result[f"{axis_name}_peaks"] = len(peaks)
    result[f"{axis_name}_zero_crossings"] = np.sum(signal[:-1] * signal[1:] < 0)

    result[f"{axis_name}_energy"] = np.sum(fft_vals**2)
    result[f"{axis_name}_dominant_freq"] = np.argmax(fft_vals)
    prob = fft_vals / np.sum(fft_vals)
    result[f"{axis_name}_entropy"] = entropy(prob)
    hist, _ = np.histogram(signal, bins=10, density=True)
    result[f"{axis_name}_signal_entropy"] = entropy(hist + 1e-10)
    return result


def calculate_3d_features(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> dict:
    result = {}
    smv = signal_magnitude_vector(np.column_stack((x, y, z)))
    x_mean, y_mean, z_mean = np.mean(x), np.mean(y), np.mean(z)

    pos_x, pos_y, pos_z, velo_acc, pos_acc = orientation(x, y, z)
    result.update(pos_x=pos_x, pos_y=pos_y, pos_z=pos_z, velo_acc=velo_acc, pos_acc=pos_acc)

    result["pitch"] = np.degrees(np.arctan2(-x_mean, np.sqrt(y_mean**2 + z_mean**2)))
    result["roll"] = np.degrees(np.arctan2(y_mean, z_mean))
    result["total_sqrt"] = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)
    result["total"] = x_mean + y_mean + z_mean
    result["signal_magnitude"] = np.sum(x + y + z) / len(x)
    result["signal_magnitude_abs"] = np.sum(np.abs(x) + np.abs(y) + np.abs(z)) / len(x)

    result["axis_corr_x_y"] = np.corrcoef(x, y)[0, 1]
    result["axis_corr_x_z"] = np.corrcoef(x, z)[0, 1]
    result["axis_corr_y_z"] = np.corrcoef(y, z)[0, 1]
    result["axis_cov_x_y"] = np.cov(x, y)[0, 1]
    result["axis_cov_x_z"] = np.cov(x, z)[0, 1]
    result["axis_cov_y_z"] = np.cov(y, z)[0, 1]

    norm = np.sqrt(x_mean**2 + y_mean**2 + z_mean**2)
    result["tilt_angle"] = np.degrees(np.arccos(z_mean / (norm + 1e-6)))
    result["spectral_entropy"] = spectral_entropy(smv)
    result["min_smv"] = np.min(smv)
    result["max_smv"] = np.max(smv)
    result["ptp_smv"] = np.ptp(smv)
    result["iqr_smv"] = iqr(smv)
    result["std_smv"] = np.std(smv)
    result["skew_smv"] = skew(smv)
    result["kurtosis_smv"] = kurtosis(smv)

    mob, comp = hjorth_parameters(smv)
    result["mobility"], result["complexity"] = mob, comp
    result["mean_crossing"] = mean_crossing_rate(smv)
    result["differential_entropy"] = differential_entropy(smv)
    result["petrosian_fd"] = petrosian_fd(smv)
    result["katz_fd"] = katz_fd(smv)
    return result


def features_for_window(window: np.ndarray) -> dict:
    """window: (150, 6) array, channel order [acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z]."""
    features = {}
    for i, name in enumerate(CHANNEL_NAMES):
        features.update(calculate_1d_features(window[:, i], name))

    acc_x, acc_y, acc_z = window[:, 0], window[:, 1], window[:, 2]
    gyr_x, gyr_y, gyr_z = window[:, 3], window[:, 4], window[:, 5]
    features.update({f"{k}_acc": v for k, v in calculate_3d_features(acc_x, acc_y, acc_z).items()})
    features.update({f"{k}_gyr": v for k, v in calculate_3d_features(gyr_x, gyr_y, gyr_z).items()})
    return features


# ============================================================
# Feature dataframe (cached per position, or per arbitrary in-memory array)
# ============================================================

def build_feature_dataframe_from_arrays(X: np.ndarray, activity: np.ndarray, subject: np.ndarray) -> pd.DataFrame:
    """Same per-window feature computation as build_feature_dataframe, but for
    an arbitrary in-memory array -- e.g. a synthetic/rotated or subject-mixed
    dataset that never lives in windows.npz."""
    rows = []
    for i in range(len(X)):
        row = features_for_window(X[i])
        row["subject"] = subject[i]
        row["activity"] = activity[i]
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_dataframe(position: str) -> pd.DataFrame:
    npz = np.load(common.WINDOWS_PATH, allow_pickle=True)
    mask = npz["position"] == position
    X, activity, subject = npz["X"][mask], npz["activity"][mask], npz["subject"][mask]
    return build_feature_dataframe_from_arrays(X, activity, subject)


def get_or_build_features_from_arrays(
    X: np.ndarray, activity: np.ndarray, subject: np.ndarray, cache_name: str,
) -> pd.DataFrame:
    cache_path = Path(__file__).parent / "processed" / f"lightgbm_features_{cache_name}.csv"
    if cache_path.exists():
        print(f"Loading cached features: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"Computing features for {cache_name} (no cache found)...")
    df = build_feature_dataframe_from_arrays(X, activity, subject)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    print(f"  {len(df)} windows, {df.shape[1] - 2} features -> cached to {cache_path}")
    return df


def get_or_build_features(position: str) -> pd.DataFrame:
    npz = np.load(common.WINDOWS_PATH, allow_pickle=True)
    mask = npz["position"] == position
    X, activity, subject = npz["X"][mask], npz["activity"][mask], npz["subject"][mask]
    return get_or_build_features_from_arrays(X, activity, subject, cache_name=position)


# ============================================================
# 5-fold leave-3-subjects-out training / evaluation
# ============================================================

def fit_lgbm(train_df: pd.DataFrame):
    """Fit once, so a caller needing multiple test sets (e.g. test_on_real_A
    and test_on_assigned) doesn't have to retrain the model for each one."""
    activity_to_label = {a: i for i, a in enumerate(common.ACTIVITIES)}
    drop_cols = ["subject", "activity"]
    x_train, y_train = train_df.drop(columns=drop_cols), train_df["activity"].map(activity_to_label)

    model = LGBMClassifier(
        subsample=0.9, boosting_type="gbdt", n_estimators=400, max_depth=15,
        learning_rate=0.1, colsample_bytree=1, n_jobs=-1, min_split_gain=0.05,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
    )
    model.fit(x_train, y_train)
    return model, activity_to_label


def predict_lgbm(model, activity_to_label: dict, test_df: pd.DataFrame):
    drop_cols = ["subject", "activity"]
    x_test, y_test = test_df.drop(columns=drop_cols), test_df["activity"].map(activity_to_label)
    y_pred = model.predict(x_test)
    return y_test.to_numpy(), y_pred


def train_and_evaluate(train_df: pd.DataFrame, test_df: pd.DataFrame):
    model, activity_to_label = fit_lgbm(train_df)
    return predict_lgbm(model, activity_to_label, test_df)


def run_cv(position: str = "forearm"):
    print(f"[lightgbm] position={position}")
    df = get_or_build_features(position)
    folds = common.load_folds()

    fold_metrics = {}
    all_yt, all_yp = [], []

    for fold in folds:
        fold_id = fold["fold"]
        train_df = df[df["subject"].isin(fold["train_subjects"])]
        test_df = df[df["subject"].isin(fold["test_subjects"])]

        yt, yp = train_and_evaluate(train_df, test_df)
        metrics = common.pooled_metrics(yt, yp)
        print(f"  fold {fold_id} (test={fold['test_subjects']}): "
              f"acc={metrics['accuracy']:.3f}  macro_f1={metrics['macro_f1']:.3f}")

        fold_metrics[fold_id] = {"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}
        all_yt.extend(yt.tolist())
        all_yp.extend(yp.tolist())

    accs = np.array([v["accuracy"] for v in fold_metrics.values()])
    f1s = np.array([v["macro_f1"] for v in fold_metrics.values()])
    pooled = common.pooled_metrics(all_yt, all_yp)

    print(f"\n====== lightgbm CV SUMMARY (position={position}) ======")
    print(f"Per-fold accuracy:  {accs.mean():.3f} +/- {accs.std():.3f}")
    print(f"Per-fold macro-F1:  {f1s.mean():.3f} +/- {f1s.std():.3f}")
    print(f"Pooled accuracy={pooled['accuracy']:.3f}  pooled macro_f1={pooled['macro_f1']:.3f}")
    return {"fold_metrics": fold_metrics, "pooled": pooled}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--position", default="forearm", choices=["chest", "forearm", "head", "shin", "thigh", "upperarm", "waist"])
    args = parser.parse_args()

    run_cv(position=args.position)

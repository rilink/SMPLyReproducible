"""
Synthesize placement-target accelerometer/gyroscope data from real placement-A
recordings, using the SMPL-derived rotation between two placements defined in
forearm_placements.yaml. sensors[0] is always A (the reference). Any other
sensor in the list (index 1 = B, index 2 = C, ...) can be used as the synthesis
target.

Same rotation applies identically to accelerometer and gyroscope channels.
Exact for gyroscope always, exact for accelerometer only when A and the
target share the same position (generally not the case here -- un-modeled
lever-arm residual otherwise).
"""

import sys
from pathlib import Path

import numpy as np
import yaml

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parents[1]
TOOL_DIR = REPO_ROOT / "tool"
MODELS_DIR = EXPERIMENTS_DIR.parent / "models"

sys.path.insert(0, str(TOOL_DIR))
sys.path.insert(0, str(MODELS_DIR))

from transform import _sensor_pos_and_frame  # noqa: E402  (reuse the exact same geometry code)
from smpl_loader import load_smpl, build_avatar, DEFAULT_SMPL, DEFAULT_BETAS  # noqa: E402
import common  # noqa: E402  (WINDOWS_PATH, ACTIVITIES, etc.)

PLACEMENTS_PATH = EXPERIMENTS_DIR / "placements" / "forearm_placements.yaml"


def target_tag(target_index: int = 1) -> str:
    """Short, content-derived tag identifying a target placement (vertex +
    z-orientation), so cached/checkpointed artifacts for different targets never
    collide.
    """
    cfg = yaml.safe_load(PLACEMENTS_PATH.read_text())
    s = cfg["sensors"][target_index]
    z_deg = s.get("orientation", {}).get("euler_xyz_deg", [0, 0, 0])[2]
    return f"{s.get('name', f'sensor{target_index}')}_v{s['vertex_idx']}-{z_deg:g}"


def target_angle_deg(target_index: int = 1, context=None) -> float:
    """Angular distance from A to the sensor at `target_index`, read live from
    forearm_placements.yaml (never hardcode a target's vertex/twist elsewhere --
    it drifts out of sync whenever the YAML entry changes)."""
    cfg = yaml.safe_load(PLACEMENTS_PATH.read_text())
    _, angle_deg = _rotation_and_angle(cfg["sensors"][target_index], context=context)
    return angle_deg


def load_A_context():
    """Load the SMPL mesh once and return (verts, tm_mesh, joints, R_A) -- reused
    across many rotation queries so the (expensive) model load only happens once.
    """
    cfg = yaml.safe_load(PLACEMENTS_PATH.read_text())
    smpl_path = TOOL_DIR / cfg.get("smpl_path", DEFAULT_SMPL)
    betas = cfg.get("betas", DEFAULT_BETAS)

    model = load_smpl(str(smpl_path))
    verts, joints, tm_mesh = build_avatar(model, betas)
    _, R_A = _sensor_pos_and_frame(cfg["sensors"][0], verts, tm_mesh, joints)
    return verts, tm_mesh, joints, R_A


def _rotation_and_angle(sensor: dict, context=None) -> tuple[np.ndarray, float]:
    """Shared core: given a sensor dict (vertex_idx + orientation), return
    (R_A_to_sensor, angular_distance_deg) against the loaded A context.
    """
    from scipy.spatial.transform import Rotation

    verts, tm_mesh, joints, R_A = context if context is not None else load_A_context()
    _, R_target = _sensor_pos_and_frame(sensor, verts, tm_mesh, joints)

    angle_deg = float(np.degrees(Rotation.from_matrix(R_A.T @ R_target).magnitude()))
    R_A_to_target = R_target.T @ R_A  # A -> target (see load_rotation_A_to_target for the derivation)
    return R_A_to_target, angle_deg


def rotation_A_to_vertex(vertex_idx: int, euler_z_deg: float, context=None) -> tuple[np.ndarray, float]:
    """Return (R, angular_distance_deg) for A -> a sensor at `vertex_idx` with
    the given Z (in-plane) twist. `context` is the (verts, tm_mesh, joints, R_A)
    tuple from `load_A_context()`. Pass it in when calling this in a loop (e.g. an
    angle sweep) to avoid reloading the SMPL model every time.
    """
    sensor = {"vertex_idx": vertex_idx, "orientation": {"euler_xyz_deg": [0, 0, euler_z_deg]}}
    return _rotation_and_angle(sensor, context=context)


def rotation_A_to_file(yaml_path, context=None) -> tuple[np.ndarray, float, str]:
    """Return (R, angular_distance_deg, tag) for A -> the first sensor found in
    a standalone single-sensor placement YAML, exactly as saved by
    `place_sensors.py` before it's been merged into forearm_placements.yaml.
    Lets you preview/use a brand-new placement immediately, without a manual
    merge step first.
    """
    sensor = yaml.safe_load(Path(yaml_path).read_text())["sensors"][0]
    R, angle_deg = _rotation_and_angle(sensor, context=context)
    z_deg = sensor.get("orientation", {}).get("euler_xyz_deg", [0, 0, 0])[2]
    tag = f"{Path(yaml_path).stem}_v{sensor['vertex_idx']}-{z_deg:g}"
    return R, angle_deg, tag


def load_rotation_A_to_target(target_index: int = 1) -> np.ndarray:
    """Return R such that x_target = R @ x_A -- apply to real A data to
    synthesize the sensor at `target_index` (sensors[0] is always A).
    """
    cfg = yaml.safe_load(PLACEMENTS_PATH.read_text())
    sensors = cfg["sensors"]
    verts, tm_mesh, joints, R_A = load_A_context()
    _, R_target = _sensor_pos_and_frame(sensors[target_index], verts, tm_mesh, joints)

    # R_rel (A <- target) = R_A.T @ R_target maps x_target -> x_A (see transform.py).
    # We want the reverse direction here: A -> target, i.e. R_rel.T = R_target.T @ R_A.
    return R_target.T @ R_A


def rotate_windows(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """X: (..., 6) with channel order [acc_x,acc_y,acc_z,gyr_x,gyr_y,gyr_z].
    Applies R independently to the acc and gyr 3-channel blocks."""
    acc = X[..., 0:3] @ R.T
    gyr = X[..., 3:6] @ R.T
    return np.concatenate([acc, gyr], axis=-1)


def build_synthetic_dataset_for_rotation(R: np.ndarray, position: str = "forearm") -> dict:
    """Load real windows at `position` and rotate them by an arbitrary, already-
    computed rotation matrix R (A -> target). This is the generic building block.
    `build_synthetic_B_dataset` is a thin YAML-driven convenience wrapper over it.
    """
    npz = np.load(common.WINDOWS_PATH, allow_pickle=True)
    mask = npz["position"] == position
    X_target = rotate_windows(npz["X"][mask].astype(np.float32), R)
    return {
        "X": X_target.astype(np.float32),
        "activity": npz["activity"][mask],
        "subject": npz["subject"][mask],
        "position": npz["position"][mask],  # kept as "forearm" -- same real position, synthetic target orientation
        "start_time_s": npz["start_time_s"][mask],
    }


def build_synthetic_B_dataset(position: str = "forearm", target_index: int = 1) -> dict:
    """Load real windows at `position`, rotate them into a synthetic version at
    the placement given by `target_index` (1 = B, 2 = C, ...). Same
    subjects/activities/labels as the real data -- only the acc/gyr values change.
    """
    R = load_rotation_A_to_target(target_index)
    return build_synthetic_dataset_for_rotation(R, position=position)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--target-index", type=int, default=1, help="1=B, 2=C, ...")
    args = parser.parse_args()

    R = load_rotation_A_to_target(args.target_index)
    tag = target_tag(args.target_index)
    print(f"Rotation A -> {tag}:")
    print(np.round(R, 5))

    data = build_synthetic_B_dataset(target_index=args.target_index)
    out_dir = EXPERIMENTS_DIR / "processed"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"synthetic_forearm_{tag}.npz"
    np.savez_compressed(out_path, **data)
    print(f"\nSaved {data['X'].shape[0]} synthetic windows -> {out_path}")

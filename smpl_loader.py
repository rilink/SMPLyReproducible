"""Shared SMPL loading utilities and constants."""

import pickle
import sys
import types
import warnings

import numpy as np
import trimesh

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── constants ─────────────────────────────────────────────────────────────────

# Override per-YAML with a top-level `betas:` list.
DEFAULT_BETAS = [1, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]
JOINT_IDX = {name: i for i, name in enumerate(JOINT_NAMES)}

SENSOR_COLORS = {
    "imu":     [0.86, 0.15, 0.25],
    "camera":  [0.20, 0.55, 1.00],
    "emg":     [0.20, 0.80, 0.45],
    "ppg":     [0.65, 0.20, 0.88],
    "default": [0.65, 0.65, 0.65],
}

DEFAULT_SMPL = "smpl/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl"

# ── SMPL loader (no chumpy dependency) ───────────────────────────────────────

def load_smpl(path):
    """Load SMPL .pkl without requiring chumpy."""
    class _Ch:
        def __init__(self, *a): self._a = a; self._s = {}
        def __setstate__(self, s): self._s = s if isinstance(s, dict) else {}
        def to_numpy(self):
            for k in ('r', 'x', 'v'):
                if k in self._s and isinstance(self._s[k], np.ndarray):
                    return np.asarray(self._s[k])
            if self._a and isinstance(self._a[0], np.ndarray):
                return np.asarray(self._a[0])
            return None

    _mod = types.ModuleType('chumpy')
    _mod.Ch = _Ch
    for n in ('chumpy', 'chumpy.ch', 'chumpy.reordering'):
        sys.modules[n] = _mod

    class _Unpickler(pickle.Unpickler):
        def find_class(self, mod, name):
            return _Ch if mod.startswith('chumpy') else super().find_class(mod, name)

    with open(path, 'rb') as f:
        raw = _Unpickler(f, encoding='latin1').load()

    return {k: np.asarray(v.to_numpy()) if isinstance(v, _Ch) and v.to_numpy() is not None else v
            for k, v in raw.items()}


# ── shape & proportion helpers ────────────────────────────────────────────────

def apply_shape(model, betas=None):
    """Return (v_shaped, joints) after blending shape parameters into v_template."""
    v = model["v_template"]
    if betas is None:
        return v, model.get("J")

    betas = np.asarray(betas, dtype=float)
    sd    = model.get("shapedirs")
    if sd is None:
        return v, model.get("J")

    sd = np.asarray(sd, dtype=float)
    if sd.ndim == 3:                                        # (V, 3, K)
        K = sd.shape[2]
        b = np.zeros(K); b[:min(len(betas), K)] = betas[:K]
        v_shaped = v + np.einsum("ijk,k->ij", sd, b)
    elif sd.ndim == 2 and sd.shape[0] == v.shape[0] * 3:   # (V*3, K)
        K = sd.shape[1]
        b = np.zeros(K); b[:min(len(betas), K)] = betas[:K]
        v_shaped = v + (sd @ b).reshape(-1, 3)
    else:
        return v, model.get("J")

    Jr = model.get("J_regressor")
    joints = np.asarray(Jr @ v_shaped) if Jr is not None else model.get("J")
    return v_shaped, joints


def build_avatar(model, betas=None):
    """Apply shape betas and return (verts, joints, trimesh)."""
    verts, joints = apply_shape(model, betas)
    tm_mesh       = build_trimesh(model, verts)
    return verts, joints, tm_mesh


def build_trimesh(model, verts=None):
    if verts is None:
        verts = model['v_template']
    faces = model['f'].astype(np.int32)
    return trimesh.Trimesh(verts, faces, process=False)


# ── geometry ──────────────────────────────────────────────────────────────────

def local_frame_from_normal(normal):
    """Right-handed frame with Z = outward normal, X = up-tangent, Y = Z×X."""
    z = np.array(normal, dtype=float)
    z /= np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    x  = up - np.dot(up, z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0]) - np.dot(np.array([1.0, 0.0, 0.0]), z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def surface_pos_and_frame(tm_mesh, query_pos):
    """Return (surface_point, R_local) for query_pos."""
    import trimesh.proximity
    pts, _, face_ids = trimesh.proximity.closest_point(tm_mesh, query_pos[np.newaxis])
    pos    = pts[0]
    normal = tm_mesh.face_normals[face_ids[0]]
    return pos, local_frame_from_normal(normal)

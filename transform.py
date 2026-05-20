"""
Compute the relative rotation between two placed sensors.

Given sensor A (reference placement) and sensor B (actual placement),
outputs the rotation R such that:

    x_in_A_frame = R @ x_in_B_frame

i.e. apply R to measurements recorded at placement B to express them
as if the sensor had been placed at A.

Usage:
    python transform.py my_setup.yaml           # interactive: click A then B
    python transform.py my_setup.yaml 0 2       # non-interactive: transform sensor 2 -> sensor 0
"""

import argparse
import sys

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from smpl_loader import (load_smpl, build_avatar, local_frame_from_normal,
                         surface_pos_and_frame, JOINT_IDX,
                         DEFAULT_SMPL, DEFAULT_BETAS)


# ── sensor geometry ───────────────────────────────────────────────────────────

def _sensor_pos_and_frame(sensor, verts, tm_mesh, joints=None):
    """Return (world_pos, R_3x3) for a sensor entry."""
    if "vertex_idx" in sensor:
        vidx = int(sensor["vertex_idx"])
        pos = verts[vidx]
        R_local = local_frame_from_normal(tm_mesh.vertex_normals[vidx])

    elif "position_3d" in sensor:
        q = np.array(sensor["position_3d"], dtype=float)
        vidx = int(np.argmin(np.linalg.norm(verts - q, axis=1)))
        pos = verts[vidx]
        R_local = local_frame_from_normal(tm_mesh.vertex_normals[vidx])

    elif "location" in sensor:
        if joints is None:
            raise ValueError("Need SMPL joints array for location-based sensors.")
        loc = sensor["location"].lower().replace(" ", "_")
        idx = JOINT_IDX.get(loc)
        if idx is None:
            raise ValueError(f"Unknown location '{sensor['location']}'.")
        pos, R_local = surface_pos_and_frame(tm_mesh, joints[idx])

    else:
        raise ValueError(f"Sensor '{sensor.get('name')}' needs vertex_idx, "
                         "position_3d, or location.")

    euler = sensor.get("orientation", {}).get("euler_xyz_deg", [0, 0, 0])
    R_user = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
    return np.asarray(pos, dtype=float), R_local @ R_user


# ── rotation output ───────────────────────────────────────────────────────────

def _compute_rotation(idx_a, idx_b, sensors, verts, tm_mesh, joints=None):
    """Return (R_rel, quat_wxyz, euler_deg, angle_deg) and print to console."""
    _, R_a = _sensor_pos_and_frame(sensors[idx_a], verts, tm_mesh, joints)
    _, R_b = _sensor_pos_and_frame(sensors[idx_b], verts, tm_mesh, joints)

    R_rel     = R_a.T @ R_b
    rot       = Rotation.from_matrix(R_rel)
    quat_xyzw = rot.as_quat()
    quat_wxyz = np.roll(quat_xyzw, 1)
    euler_deg = rot.as_euler("xyz", degrees=True)
    angle_deg = float(np.degrees(rot.magnitude()))

    print(f"\n-- Relative rotation:  [{idx_b}] {sensors[idx_b]['name']}"
          f"  ->  [{idx_a}] {sensors[idx_a]['name']} --")
    print("Apply R to measurements in B's frame to get them in A's frame.\n")
    print("Rotation matrix (R):")
    for row in R_rel:
        print("  " + "  ".join(f"{v:8.5f}" for v in row))
    print(f"\nQuaternion  w x y z : {np.round(quat_wxyz, 6)}")
    print(f"Quaternion  x y z w : {np.round(quat_xyzw, 6)}")
    print(f"Euler XYZ (degrees) : {np.round(euler_deg, 3)}")
    print(f"\nAngular distance    : {angle_deg:.2f} deg")

    return R_rel, quat_wxyz, euler_deg, angle_deg


# ── interactive picker ────────────────────────────────────────────────────────

_VEDO_TYPE_COLORS = {
    "imu":     "#db2640",
    "camera":  "#3388ff",
    "emg":     "#33cc72",
    "ppg":     "#a632e0",
    "default": "#aaaaaa",
}
_AXIS_COLORS  = ["#e02020", "#20c050", "#2060e0"]
_FRAME_SCALE  = 0.09
_SPHERE_R     = 0.024
_COL_A        = "#f5a00a"   # amber  — reference sensor A
_COL_B        = "#2255cc"   # blue   — sensor B


def _make_glyph(pos, R_frame, stype, plt):
    """Add sphere + RGB frame sticks to plotter; return (sphere, [sticks])."""
    from vedo import Sphere as VSphere, Cylinder as VCylinder
    col    = _VEDO_TYPE_COLORS.get(stype, _VEDO_TYPE_COLORS["default"])
    sphere = VSphere(pos, r=_SPHERE_R, c=col, alpha=1.0)
    sticks = [
        VCylinder(pos=[pos.tolist(), (pos + R_frame[:, i] * _FRAME_SCALE).tolist()],
                  r=0.004, c=_AXIS_COLORS[i], cap=True)
        for i in range(3)
    ]
    plt.add(sphere, *sticks)
    return sphere, sticks


def _interactive(sensors, verts, faces, tm_mesh, joints=None):
    from vedo import Plotter, Mesh as VMesh, Text3D, Text2D

    # resolve positions + frames once
    poses, frames = [], []
    for s in sensors:
        try:
            pos, R = _sensor_pos_and_frame(s, verts, tm_mesh, joints)
        except ValueError as e:
            print(f"  [WARN] {e}  — using origin as placeholder")
            pos, R = np.zeros(3), np.eye(3)
        poses.append(pos)
        frames.append(R)

    state = {"sel": [], "type_cols": []}

    plt = Plotter(title="SMPLy Reproducible -- Select sensor A then B",
                  bg="white", size=(960, 900))

    body = VMesh([verts, faces], c="#acbcd2", alpha=0.85)
    body.compute_normals()
    plt.add(body)

    spheres = []
    for i, (s, pos, R) in enumerate(zip(sensors, poses, frames)):
        stype = s.get("type", "default").lower()
        sphere, _ = _make_glyph(pos, R, stype, plt)
        spheres.append(sphere)
        state["type_cols"].append(_VEDO_TYPE_COLORS.get(stype, _VEDO_TYPE_COLORS["default"]))

        lpos = (pos + np.array([0.05, 0.03, 0.0])).tolist()
        plt.add(Text3D(f"[{i}] {s['name']}", pos=lpos, s=0.011, c="k2"))

    hud = Text2D(
        "Click sensor A (reference), then sensor B\n"
        "R = reset   Q = quit",
        pos="top-left", s=0.82, c="k4", bg="w8", font="Calco"
    )
    plt.add(hud)

    def _reset():
        state["sel"].clear()
        for sp, col in zip(spheres, state["type_cols"]):
            sp.c(col)
        hud.text("Click sensor A (reference), then sensor B\nR = reset   Q = quit")
        plt.render()

    def on_click(evt):
        if evt.picked3d is None:
            return
        cp = np.array(evt.picked3d, dtype=float)

        best_i, best_d = -1, 0.08
        for i, pos in enumerate(poses):
            d = float(np.linalg.norm(pos - cp))
            if d < best_d:
                best_d = d
                best_i = i
        if best_i < 0:
            return

        sel = state["sel"]
        if len(sel) == 0:
            sel.append(best_i)
            spheres[best_i].c(_COL_A)
            hud.text(
                f"A = [{best_i}] {sensors[best_i]['name']}\n"
                "Now click sensor B\n"
                "R = reset   Q = quit"
            )
            plt.render()

        elif len(sel) == 1:
            if best_i == sel[0]:
                return
            sel.append(best_i)
            spheres[best_i].c(_COL_B)
            plt.render()

            R_rel, quat_wxyz, euler_deg, angle_deg = _compute_rotation(
                sel[0], sel[1], sensors, verts, tm_mesh, joints
            )
            hud.text(
                f"A (ref) = [{sel[0]}] {sensors[sel[0]]['name']}\n"
                f"B       = [{sel[1]}] {sensors[sel[1]]['name']}\n"
                f"Euler XYZ: {np.round(euler_deg, 1)} deg\n"
                f"Angular dist: {angle_deg:.1f} deg\n"
                "R = reset   Q = quit"
            )
            plt.render()

    def on_key(evt):
        if evt.keypress.lower() == "r":
            _reset()

    plt.add_callback("LeftButtonPress", on_click)
    plt.add_callback("key press",       on_key)

    print("\nSensor list:")
    for i, s in enumerate(sensors):
        euler = s.get("orientation", {}).get("euler_xyz_deg", [0, 0, 0])
        print(f"  [{i}]  {s['name']:30s}  type={s.get('type','?')}  "
              f"z_rot={euler[2]:+.0f}deg")
    print("\nClick sensor A in the 3-D window, then sensor B.")
    print("Rotation result will be printed here and shown in the HUD.\n")

    plt.show(viewup="y", zoom=1.2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config",              help="Sensor placement YAML")
    parser.add_argument("sensor_a", nargs="?", type=int,
                        help="Index of reference sensor A  (omit for interactive mode)")
    parser.add_argument("sensor_b", nargs="?", type=int,
                        help="Index of sensor B to transform into A")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sensors   = cfg.get("sensors", [])
    smpl_path = cfg.get("smpl_path", DEFAULT_SMPL)

    betas  = cfg.get("betas", DEFAULT_BETAS)
    model  = load_smpl(smpl_path)
    verts, joints, tm_mesh = build_avatar(model, betas)
    faces  = model["f"].astype(np.int32)

    if args.sensor_a is None or args.sensor_b is None:
        _interactive(sensors, verts, faces, tm_mesh, joints)
        return

    # non-interactive: print sensor list then compute
    print(f"\nSensors in '{args.config}':")
    for i, s in enumerate(sensors):
        euler = s.get("orientation", {}).get("euler_xyz_deg", [0, 0, 0])
        print(f"  [{i}]  {s['name']:30s}  type={s.get('type','?')}  "
              f"z_rot={euler[2]:+.0f}deg")

    n = len(sensors)
    if not (0 <= args.sensor_a < n) or not (0 <= args.sensor_b < n):
        print(f"\nError: indices must be between 0 and {n - 1}.")
        sys.exit(1)

    _compute_rotation(args.sensor_a, args.sensor_b,
                      sensors, verts, tm_mesh, joints)


if __name__ == "__main__":
    main()

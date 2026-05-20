"""
Visualize wearable sensor placements on the SMPL body model.

Usage:
    python visualize.py sensors_example.yaml
    python visualize.py sensors_placed.yaml    # output from place_sensors.py

Sensor position can be specified as:
    location: right_wrist        # SMPL joint name (see smpl_loader.JOINT_NAMES)
    vertex_idx: 4523             # exact vertex index (from place_sensors.py)
    position_3d: [x, y, z]      # nearest-vertex lookup

Sensor type controls marker color:
    imu -> red  |  camera -> blue  |  emg -> green  |  ppg -> purple

orientation.euler_xyz_deg: [X, Y, Z] degrees, in the sensor's local surface frame.
  [0, 0, 0] = sensor flat on skin, Z pointing outward from surface.

Keyboard controls:
    up / down arrows   body opacity
    7 / 8              background (white / dark)
    k                  shiny surface
    q                  quit
"""

import argparse
import warnings

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from vedo import Plotter, Mesh as VMesh, Sphere as VSphere, Cylinder as VCylinder, Text2D

from smpl_loader import (load_smpl, build_avatar, local_frame_from_normal,
                         surface_pos_and_frame, JOINT_IDX, JOINT_NAMES,
                         DEFAULT_SMPL, DEFAULT_BETAS)

warnings.filterwarnings("ignore")

_VEDO_COLORS = {
    "imu":     "#db2640",
    "camera":  "#3388ff",
    "emg":     "#33cc72",
    "ppg":     "#a632e0",
    "default": "#aaaaaa",
}

FRAME_SCALE = 0.10
SPHERE_R    = 0.022


def _resolve_position(sensor, verts, tm_mesh, joints):
    if "vertex_idx" in sensor:
        vidx   = int(sensor["vertex_idx"])
        pos    = verts[vidx]
        normal = tm_mesh.vertex_normals[vidx]
        return pos, local_frame_from_normal(normal)

    if "position_3d" in sensor:
        query = np.array(sensor["position_3d"], dtype=float)
        vidx  = int(np.argmin(np.linalg.norm(verts - query, axis=1)))
        pos   = verts[vidx]
        return pos, local_frame_from_normal(tm_mesh.vertex_normals[vidx])

    if "location" in sensor:
        loc = sensor["location"].lower().replace(" ", "_")
        idx = JOINT_IDX.get(loc)
        if idx is None:
            raise ValueError(f"Unknown location '{sensor['location']}'. "
                             f"Valid: {', '.join(JOINT_NAMES)}")
        return surface_pos_and_frame(tm_mesh, joints[idx])

    raise ValueError(f"Sensor '{sensor.get('name')}' needs location, "
                     "vertex_idx, or position_3d.")


def _sensor_actors(pos, R_frame, stype):
    """Sphere + 3 axis sticks — identical glyph style to place_sensors.py."""
    col    = _VEDO_COLORS.get(stype, _VEDO_COLORS["default"])
    sphere = VSphere(pos, r=SPHERE_R, c=col, alpha=1.0)
    axis_colors = ["#e02020", "#20c050", "#2060e0"]
    sticks = [
        VCylinder(pos=[pos.tolist(), (pos + R_frame[:, i] * FRAME_SCALE).tolist()],
                  r=0.004, c=axis_colors[i], cap=True)
        for i in range(3)
    ]
    return [sphere] + sticks


_CAMERAS = {
    "front": dict(pos=(0, 0.35, 2.8), focalPoint=(0, 0.35, 0), viewup=(0, 1, 0)),
    "back":  dict(pos=(0, 0.35,-2.8), focalPoint=(0, 0.35, 0), viewup=(0, 1, 0)),
}


def visualize(config_path, view=None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    smpl_path = cfg.get("smpl_path", DEFAULT_SMPL)
    betas  = cfg.get("betas", DEFAULT_BETAS)
    model  = load_smpl(smpl_path)
    verts, joints, tm_mesh = build_avatar(model, betas)
    faces  = model["f"].astype(np.int32)

    plt = Plotter(title="SMPLy Reproducible -- Sensor Placement",
                  bg="white", size=(900, 900))

    body = VMesh([verts, faces], c="#acbcd2", alpha=0.3)
    body.compute_normals()
    plt.add(body)

    sensor_list = cfg.get("sensors", [])
    if not sensor_list:
        print("No sensors defined — showing body only.")
    else:
        for sensor in sensor_list:
            name  = sensor.get("name", "sensor")
            stype = sensor.get("type", "default").lower()
            euler = sensor.get("orientation", {}).get("euler_xyz_deg", [0, 0, 0])

            try:
                pos, R_local = _resolve_position(sensor, verts, tm_mesh, joints)
            except ValueError as e:
                print(f"  [WARN] {e}")
                continue

            R_user = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
            rotmat = R_local @ R_user

            plt.add(*_sensor_actors(pos, rotmat, stype))
            print(f"  [{name}]  type={stype}  pos={np.round(pos, 3)}")

    hud = Text2D(
        f" {len(sensor_list)} sensor(s) from {config_path}\n"
        f" ────────────────────────────\n"
        f" ↑↓ opacity  |  7/8 background\n"
        f" k  shiny surface\n"
        f" q  quit",
        pos="top-left", s=0.78, c="k4", bg="w8", font="Calco"
    )
    plt.add(hud)

    print(f"\nViewing {len(sensor_list)} sensor(s). Press Q to quit.")
    cam = _CAMERAS.get(view)
    if cam:
        plt.show(camera=cam)
    else:
        plt.show(viewup="y", zoom=1.2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", default="sensors_placed.yaml")
    parser.add_argument("--view", choices=["front", "back"],
                        help="Open with front or back camera (default: perspective)")
    args = parser.parse_args()
    visualize(args.config, view=args.view)

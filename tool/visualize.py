"""
Visualize wearable sensor placements on the SMPL body model.

Usage:
    python tool/visualize.py tool/placements/sensors_example.yaml
    python tool/visualize.py tool/placements/sensors_placed.yaml  # output from place_sensors.py

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
                         DEFAULT_SMPL, DEFAULT_BETAS, VEDO_COLORS, AXIS_COLORS,
                         CAMERAS, PLACEMENTS_DIR)

warnings.filterwarnings("ignore")

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
    col    = VEDO_COLORS.get(stype, VEDO_COLORS["default"])
    sphere = VSphere(pos, r=SPHERE_R, c=col, alpha=1.0)
    sticks = [
        VCylinder(pos=[pos.tolist(), (pos + R_frame[:, i] * FRAME_SCALE).tolist()],
                  r=0.004, c=AXIS_COLORS[i], cap=True)
        for i in range(3)
    ]
    return [sphere] + sticks


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

    body = VMesh([verts, faces], c="#acbcd2", alpha=1.0)
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
    cam = CAMERAS.get(view)
    if cam:
        plt.show(camera=cam)
    else:
        plt.show(viewup="y", zoom=1.2)


def _build_scene(plt, cfg, verts, tm_mesh, joints, faces):
    """Shared scene-building logic: add the body mesh and every sensor's
    glyph to an already-created Plotter. Used by both the interactive
    visualize() and the offscreen render_views_to_pdf()."""
    body = VMesh([verts, faces], c="#acbcd2", alpha=1.0)
    body.compute_normals()
    plt.add(body)

    for sensor in cfg.get("sensors", []):
        stype = sensor.get("type", "default").lower()
        euler = sensor.get("orientation", {}).get("euler_xyz_deg", [0, 0, 0])
        try:
            pos, R_local = _resolve_position(sensor, verts, tm_mesh, joints)
        except ValueError as e:
            print(f"  [WARN] {e}")
            continue
        R_user = Rotation.from_euler("xyz", euler, degrees=True).as_matrix()
        plt.add(*_sensor_actors(pos, R_local @ R_user, stype))


def render_views_to_pdf(config_path, output_dir=".", prefix=None, size=(1600, 1600), scale=3):
    """Render the front and back views offscreen (no interactive window) and
    save each straight to PDF -- for dropping directly into a LaTeX figure.
    A PNG screenshot is saved alongside each PDF too, in case a raster
    preview is more convenient.
    """
    from pathlib import Path
    from PIL import Image

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    smpl_path = cfg.get("smpl_path", DEFAULT_SMPL)
    betas = cfg.get("betas", DEFAULT_BETAS)
    model = load_smpl(smpl_path)
    verts, joints, tm_mesh = build_avatar(model, betas)
    faces = model["f"].astype(np.int32)

    prefix = prefix or Path(config_path).stem
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_pdfs = []
    for view_name, cam in CAMERAS.items():
        plt = Plotter(offscreen=True, bg="white", size=size)
        _build_scene(plt, cfg, verts, tm_mesh, joints, faces)
        plt.show(camera=cam, interactive=False)
        # fixed camera dict skips vedo auto-fit; refit bounds, keep direction
        plt.renderer.ResetCamera()
        plt.render()

        png_path = output_dir / f"{prefix}_{view_name}.png"
        # scale: supersample before downsampling, sharper than flat screenshot
        plt.screenshot(str(png_path), scale=scale)
        plt.close()

        pdf_path = output_dir / f"{prefix}_{view_name}.pdf"
        Image.open(png_path).convert("RGB").save(pdf_path, "PDF", resolution=300.0)
        saved_pdfs.append(pdf_path)
        print(f"Saved {view_name} view -> {pdf_path}  (and {png_path})")

    return saved_pdfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?",
                        default=str(PLACEMENTS_DIR / "sensors_placed.yaml"))
    parser.add_argument("--view", choices=["front", "back"],
                        help="Open with front or back camera (default: perspective)")
    parser.add_argument("--save-pdf", action="store_true",
                        help="Skip the interactive window; render front+back offscreen and save each to PDF")
    parser.add_argument("--output-dir", default=".",
                        help="Directory to save PDFs/PNGs into (used with --save-pdf)")
    args = parser.parse_args()

    if args.save_pdf:
        render_views_to_pdf(args.config, output_dir=args.output_dir)
    else:
        visualize(args.config, view=args.view)

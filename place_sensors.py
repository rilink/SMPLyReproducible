"""
Interactive sensor placement on the SMPL body model.

Left-click on the body to place a sensor at that surface point.
The coordinate frame (RGB sticks) reflects the local body surface orientation.

Controls:
  i  → set type: IMU    (red)
  c  → set type: Camera (blue)
  e  → set type: EMG    (green)
  g  → set type: PPG    (purple)
  h  → rotate last sensor -10° around blue axis
  j  → rotate last sensor +10° around blue axis
  u  → undo last placed sensor
  s  → save to YAML and quit
  q  → quit without saving

Usage:
    python place_sensors.py [output.yaml] [--smpl PATH]
"""

import argparse
import warnings

import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from vedo import Plotter, Mesh as VMesh, Sphere as VSphere, Cylinder as VCylinder, Text2D

from smpl_loader import (load_smpl, build_avatar, local_frame_from_normal,
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
ROT_STEP    = 10   # degrees per Left / Right arrow press

_TYPE_LABEL = {
    "imu": "IMU (red)", "camera": "Camera (blue)",
    "emg": "EMG (green)", "ppg": "PPG (purple)",
}


# ── sensor glyph ─────────────────────────────────────────────────────────────

def _sensor_actors(pos, R_frame, stype):
    """Sphere + 3 axis sticks. R_frame columns = X/Y/Z axes in world coords."""
    col = _VEDO_COLORS.get(stype, "#aaaaaa")
    sphere = VSphere(pos, r=SPHERE_R, c=col, alpha=1.0)
    axis_colors = ["#e02020", "#20c050", "#2060e0"]
    sticks = [
        VCylinder(pos=[pos, pos + R_frame[:, i] * FRAME_SCALE],
                  r=0.004, c=axis_colors[i], cap=True)
        for i in range(3)
    ]
    return [sphere] + sticks


def _rotated_frame(R_local, z_deg):
    """Apply z_deg rotation around the surface normal (local Z) to R_local."""
    R_z = Rotation.from_euler("z", z_deg, degrees=True).as_matrix()
    return R_local @ R_z


# ── HUD ───────────────────────────────────────────────────────────────────────

def _hud_text(stype, n_placed, z_deg):
    label = _TYPE_LABEL.get(stype, stype)
    return (
        f" Type    : {label}\n"
        f" Placed  : {n_placed}\n"
        f" Z-rot   : {z_deg:+.0f}°  (red+green spin around blue)\n"
        f" ──────────────────────────────────\n"
        f" i=IMU  c=Camera  e=EMG  g=PPG\n"
        f" h / j  rotate last sensor ±10° around blue axis\n"
        f" u=undo  |  ↑↓ opacity  |  7/8 background\n"
        f" s=save+quit   q=quit "
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", nargs="?", default="sensors_placed.yaml")
    parser.add_argument("--smpl", default=DEFAULT_SMPL)
    parser.add_argument("--view", choices=["front", "back"],
                        help="Open with front or back camera (default: perspective)")
    args = parser.parse_args()

    print(f"Loading SMPL model from '{args.smpl}' …")
    model             = load_smpl(args.smpl)
    verts, _, tm_mesh = build_avatar(model, DEFAULT_BETAS)
    faces             = model["f"].astype(np.int32)

    state = {
        "type":    "imu",
        "sensors": [],
        "actors":  [],
        "z_deg":   0,      # current Z-rotation for the last placed sensor
        "saved":   False,
    }

    plt = Plotter(title="SMPLy Reproducible — Interactive Sensor Placement",
                  bg="white", size=(960, 900))

    body = VMesh([verts, faces], c="#acbcd2", alpha=0.3)
    body.compute_normals()
    plt.add(body)

    hud = Text2D(_hud_text("imu", 0, 0),
                 pos="top-left", s=0.78, c="k4", bg="w8", font="Calco")
    plt.add(hud)

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _refresh_last(z_deg):
        """Redraw the last sensor's frame sticks with updated Z rotation."""
        if not state["sensors"]:
            print("  (no sensor to rotate yet)")
            return
        last    = state["sensors"][-1]
        pos     = np.array(last["position_3d"], dtype=float)
        normal  = tm_mesh.vertex_normals[last["vertex_idx"]]
        R_local = local_frame_from_normal(normal)
        R_frame = _rotated_frame(R_local, z_deg)

        # remove only the 3 sticks (index 1-3); keep the sphere (index 0)
        for a in state["actors"][-1][1:]:
            plt.remove(a)

        axis_colors = ["#e02020", "#20c050", "#2060e0"]
        new_sticks = [
            VCylinder(pos=[pos.tolist(), (pos + R_frame[:, i] * FRAME_SCALE).tolist()],
                      r=0.004, c=axis_colors[i], cap=True)
            for i in range(3)
        ]
        state["actors"][-1] = [state["actors"][-1][0]] + new_sticks
        plt.add(*new_sticks)
        last["orientation"]["euler_xyz_deg"] = [0, 0, round(z_deg, 1)]
        print(f"  ~ {last['name']}  z_rot={z_deg:+.0f}°")

    def on_click(evt):
        if evt.picked3d is None or evt.actor is not body:
            return
        click_pos = np.array(evt.picked3d, dtype=float)

        dists = np.linalg.norm(verts - click_pos, axis=1)
        vidx  = int(np.argmin(dists))
        pos   = verts[vidx]
        normal = tm_mesh.vertex_normals[vidx]
        R_local = local_frame_from_normal(normal)
        R_frame = _rotated_frame(R_local, state["z_deg"])

        stype = state["type"]
        count = sum(1 for s in state["sensors"] if s["type"] == stype)
        name  = f"{stype.capitalize()}_{count + 1}"

        sensor = {
            "name":        name,
            "type":        stype,
            "vertex_idx":  vidx,
            "position_3d": [round(float(v), 5) for v in pos],
            "orientation": {"euler_xyz_deg": [0, 0, round(state["z_deg"], 1)]},
        }
        state["sensors"].append(sensor)

        actors = _sensor_actors(pos, R_frame, stype)
        state["actors"].append(actors)
        plt.add(*actors)

        hud.text(_hud_text(stype, len(state["sensors"]), state["z_deg"]))
        plt.render()
        print(f"  + {name}  vertex={vidx}  z_rot={state['z_deg']:+.0f}°  "
              f"pos={np.round(pos, 3)}")

    def on_key(evt):
        k = evt.keypress
        if   k == "i": state["type"] = "imu"
        elif k == "c": state["type"] = "camera"
        elif k == "e": state["type"] = "emg"
        elif k == "g": state["type"] = "ppg"
        elif k == "h":
            state["z_deg"] = (state["z_deg"] - ROT_STEP) % 360
            _refresh_last(state["z_deg"])
        elif k == "j":
            state["z_deg"] = (state["z_deg"] + ROT_STEP) % 360
            _refresh_last(state["z_deg"])
        elif k == "u":
            if state["sensors"]:
                removed = state["sensors"].pop()
                for a in state["actors"].pop():
                    plt.remove(a)
                state["z_deg"] = removed["orientation"]["euler_xyz_deg"][2]
                print(f"  - Removed {removed['name']}")
        elif k == "s":
            _save(state["sensors"], args.output, args.smpl)
            state["saved"] = True
            plt.close()
            return

        hud.text(_hud_text(state["type"], len(state["sensors"]), state["z_deg"]))
        plt.render()

    plt.add_callback("LeftButtonPress", on_click)
    plt.add_callback("key press",       on_key)

    print("\nControls:  click=place  h/j=rotate sensor  ↑↓=opacity  7/8=background  k=shiny  s=save  q=quit\n")
    _CAMERAS = {
        "front": dict(pos=(0, 0.35, 2.8), focalPoint=(0, 0.35, 0), viewup=(0, 1, 0)),
        "back":  dict(pos=(0, 0.35,-2.8), focalPoint=(0, 0.35, 0), viewup=(0, 1, 0)),
    }
    cam = _CAMERAS.get(args.view)
    if cam:
        plt.show(camera=cam)
    else:
        plt.show(viewup="y", zoom=1.2)

    if state["sensors"] and not state["saved"]:
        print(f"\n{len(state['sensors'])} sensor(s) not saved (quit without 's').")


def _save(sensors, out_path, smpl_path):
    cfg = {"smpl_path": smpl_path, "sensors": sensors}
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved {len(sensors)} sensor(s) to '{out_path}'")
    print(f"Visualise with:  python visualize.py {out_path}")


if __name__ == "__main__":
    main()

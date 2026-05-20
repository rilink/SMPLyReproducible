"""
Place sensors at known SMPL vertex indices and adjust their orientation.

Pass one or more vertex indices on the command line.  The script places all
sensors immediately and lets you click each one to select it, then rotate its
local frame with h / j before saving.

Usage:
    python place_sensors_via_vertex.py 4523 1234 5678
    python place_sensors_via_vertex.py 4523 1234 --output my_setup.yaml --view front

Controls:
    Left-click a sensor     select it  (turns amber)
    h / j                   rotate selected sensor -/+10 deg around blue axis
    i / c / e / g           change type of selected sensor (IMU/Camera/EMG/PPG)
    s                       save to YAML and quit
    q                       quit without saving
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
_AXIS_COLORS  = ["#e02020", "#20c050", "#2060e0"]
_COL_SELECTED = "#f5a00a"

FRAME_SCALE = 0.10
SPHERE_R    = 0.022
ROT_STEP    = 10

_CAMERAS = {
    "front": dict(pos=(0, 0.35, 2.8), focalPoint=(0, 0.35, 0), viewup=(0, 1, 0)),
    "back":  dict(pos=(0, 0.35,-2.8), focalPoint=(0, 0.35, 0), viewup=(0, 1, 0)),
}

_TYPE_LABEL = {
    "imu": "IMU (red)", "camera": "Camera (blue)",
    "emg": "EMG (green)", "ppg": "PPG (purple)",
}


def _make_sticks(pos, R_frame):
    return [
        VCylinder(pos=[pos.tolist(), (pos + R_frame[:, i] * FRAME_SCALE).tolist()],
                  r=0.004, c=_AXIS_COLORS[i], cap=True)
        for i in range(3)
    ]


def _hud_text(sensors, sel):
    s     = sensors[sel]
    z_deg = s["orientation"]["euler_xyz_deg"][2]
    label = _TYPE_LABEL.get(s["type"], s["type"])
    return (
        f" Selected : [{sel}] {s['name']}\n"
        f" Type     : {label}\n"
        f" Z-rot    : {z_deg:+.0f} deg  (red+green around blue)\n"
        f" ────────────────────────────────────\n"
        f" click sensor to select\n"
        f" h / j   rotate selected +-10 deg around blue axis\n"
        f" i=IMU  c=Camera  e=EMG  g=PPG\n"
        f" s=save+quit   q=quit"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vertex_ids", nargs="+", type=int,
                        help="SMPL vertex indices")
    parser.add_argument("--output", default="sensors_placed.yaml")
    parser.add_argument("--smpl",   default=DEFAULT_SMPL)
    parser.add_argument("--view",   choices=["front", "back"])
    args = parser.parse_args()

    print(f"Loading SMPL model from '{args.smpl}' ...")
    model             = load_smpl(args.smpl)
    verts, _, tm_mesh = build_avatar(model, DEFAULT_BETAS)
    faces             = model["f"].astype(np.int32)
    n_verts           = len(verts)

    # ── build sensor list from given vertex IDs ───────────────────────────────
    sensors = []
    for i, vidx in enumerate(args.vertex_ids):
        if not (0 <= vidx < n_verts):
            print(f"  [WARN] vertex_idx {vidx} out of range (0 – {n_verts - 1}) — skipped")
            continue
        pos = verts[vidx]
        sensors.append({
            "name":        f"Imu_{i + 1}",
            "type":        "imu",
            "vertex_idx":  vidx,
            "position_3d": [round(float(v), 5) for v in pos],
            "orientation": {"euler_xyz_deg": [0, 0, 0]},
        })
        print(f"  [{i}]  vertex_idx={vidx}  pos={np.round(pos, 3)}")

    if not sensors:
        print("No valid vertex indices — exiting.")
        return

    state = {"sel": 0, "saved": False}

    plt = Plotter(title="SMPLy Reproducible — Sensor Placement via Vertex ID",
                  bg="white", size=(960, 900))

    body = VMesh([verts, faces], c="#acbcd2", alpha=0.3)
    body.compute_normals()
    plt.add(body)

    # ── place sensor glyphs ───────────────────────────────────────────────────
    def _frame(s):
        normal  = tm_mesh.vertex_normals[s["vertex_idx"]]
        R_local = local_frame_from_normal(normal)
        z_deg   = s["orientation"]["euler_xyz_deg"][2]
        return R_local @ Rotation.from_euler("z", z_deg, degrees=True).as_matrix()

    actors = []   # each entry: [sphere, stick0, stick1, stick2]
    for s in sensors:
        pos    = np.array(s["position_3d"], dtype=float)
        col    = _VEDO_COLORS.get(s["type"], "#aaaaaa")
        sphere = VSphere(pos, r=SPHERE_R, c=col, alpha=1.0)
        sticks = _make_sticks(pos, _frame(s))
        plt.add(sphere, *sticks)
        actors.append([sphere] + sticks)

    # highlight first sensor
    actors[0][0].c(_COL_SELECTED)

    hud = Text2D(_hud_text(sensors, 0),
                 pos="top-left", s=0.78, c="k4", bg="w8", font="Calco")
    plt.add(hud)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _redraw_sticks(idx):
        s   = sensors[idx]
        pos = np.array(s["position_3d"], dtype=float)
        for stick in actors[idx][1:]:
            plt.remove(stick)
        new_sticks = _make_sticks(pos, _frame(s))
        actors[idx] = [actors[idx][0]] + new_sticks
        plt.add(*new_sticks)

    def _select(idx):
        # restore previous sphere colour
        prev = state["sel"]
        actors[prev][0].c(_VEDO_COLORS.get(sensors[prev]["type"], "#aaaaaa"))
        state["sel"] = idx
        actors[idx][0].c(_COL_SELECTED)
        hud.text(_hud_text(sensors, idx))
        plt.render()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def on_click(evt):
        if evt.picked3d is None:
            return
        cp = np.array(evt.picked3d, dtype=float)
        best_i, best_d = -1, 0.08
        for i, s in enumerate(sensors):
            d = float(np.linalg.norm(np.array(s["position_3d"]) - cp))
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0 and best_i != state["sel"]:
            _select(best_i)

    def on_key(evt):
        k   = evt.keypress
        idx = state["sel"]
        s   = sensors[idx]

        if k == "h":
            s["orientation"]["euler_xyz_deg"][2] = (
                s["orientation"]["euler_xyz_deg"][2] - ROT_STEP) % 360
            _redraw_sticks(idx)
            print(f"  ~ [{idx}] {s['name']}  vertex_idx={s['vertex_idx']}"
                  f"  z_rot={s['orientation']['euler_xyz_deg'][2]:+.0f} deg")

        elif k == "j":
            s["orientation"]["euler_xyz_deg"][2] = (
                s["orientation"]["euler_xyz_deg"][2] + ROT_STEP) % 360
            _redraw_sticks(idx)
            print(f"  ~ [{idx}] {s['name']}  vertex_idx={s['vertex_idx']}"
                  f"  z_rot={s['orientation']['euler_xyz_deg'][2]:+.0f} deg")

        elif k in ("i", "c", "e", "g"):
            t = {"i": "imu", "c": "camera", "e": "emg", "g": "ppg"}[k]
            s["type"] = t
            count = sum(1 for x in sensors if x["type"] == t)
            s["name"] = f"{t.capitalize()}_{count}"
            actors[idx][0].c(_VEDO_COLORS.get(t, "#aaaaaa"))  # show new type colour

        elif k == "s":
            _save(sensors, args.output, args.smpl)
            state["saved"] = True
            plt.close()
            return

        hud.text(_hud_text(sensors, idx))
        plt.render()

    plt.add_callback("LeftButtonPress", on_click)
    plt.add_callback("key press",       on_key)

    cam = _CAMERAS.get(args.view)
    plt.show(camera=cam) if cam else plt.show(viewup="y", zoom=1.2)

    if not state["saved"]:
        print(f"\n{len(sensors)} sensor(s) not saved (quit without 's').")


def _save(sensors, out_path, smpl_path):
    cfg = {"smpl_path": smpl_path, "sensors": sensors}
    with open(out_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved {len(sensors)} sensor(s) to '{out_path}'")
    print(f"Visualise with:  python visualize.py {out_path}")


if __name__ == "__main__":
    main()

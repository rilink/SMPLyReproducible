"""
Click on the SMPL body to inspect vertex indices.

Each left-click prints the nearest vertex index and its 3-D position to the
console and drops a small marker on the mesh.  Use this to find vertex_idx
values for manual YAML sensor configs.

Usage:
    python pick_vertex.py
    python pick_vertex.py --smpl smpl/your_model.pkl
    python pick_vertex.py --view front
"""

import argparse
import warnings

import numpy as np
from vedo import Plotter, Mesh as VMesh, Sphere as VSphere, Cylinder as VCylinder, Text2D

from smpl_loader import (load_smpl, build_avatar, local_frame_from_normal,
                         DEFAULT_SMPL, DEFAULT_BETAS)

warnings.filterwarnings("ignore")

_CAMERAS = {
    "front": dict(pos=(0, 0.35, 2.8), focalPoint=(0, 0.35, 0), viewup=(0, 1, 0)),
    "back":  dict(pos=(0, 0.35,-2.8), focalPoint=(0, 0.35, 0), viewup=(0, 1, 0)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smpl", default=DEFAULT_SMPL)
    parser.add_argument("--view", choices=["front", "back"],
                        help="Starting camera view (default: perspective)")
    args = parser.parse_args()

    print(f"Loading SMPL model from '{args.smpl}' …")
    model              = load_smpl(args.smpl)
    verts, _, tm_mesh  = build_avatar(model, DEFAULT_BETAS)
    faces              = model["f"].astype(np.int32)

    FRAME_SCALE  = 0.10
    AXIS_COLORS  = ["#e02020", "#20c050", "#2060e0"]

    plt = Plotter(title="SMPLy Reproducible -- Vertex Picker",
                  bg="white", size=(900, 900))

    body = VMesh([verts, faces], c="#acbcd2", alpha=0.3)
    body.compute_normals()
    plt.add(body)

    hud = Text2D(
        " Left-click to pick a vertex\n"
        " ────────────────────────────\n"
        " ↑↓ opacity  |  7/8 background\n"
        " q  quit",
        pos="top-left", s=0.78, c="k4", bg="w8", font="Calco"
    )
    plt.add(hud)

    def on_click(evt):
        if evt.picked3d is None or evt.actor is not body:
            return
        click_pos = np.array(evt.picked3d, dtype=float)
        vidx = int(np.argmin(np.linalg.norm(verts - click_pos, axis=1)))
        pos  = verts[vidx]

        normal  = tm_mesh.vertex_normals[vidx]
        R_frame = local_frame_from_normal(normal)

        print(f"vertex_idx: {vidx:6d}   pos: [{pos[0]:+.5f}, {pos[1]:+.5f}, {pos[2]:+.5f}]")

        marker = VSphere(pos, r=0.012, c="#f5a00a", alpha=1.0)
        sticks = [
            VCylinder(pos=[pos.tolist(), (pos + R_frame[:, i] * FRAME_SCALE).tolist()],
                      r=0.004, c=AXIS_COLORS[i], cap=True)
            for i in range(3)
        ]
        plt.add(marker, *sticks)

        hud.text(
            f" vertex_idx : {vidx}\n"
            f" pos        : [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]\n"
            f" ────────────────────────────\n"
            f" ↑↓ opacity  |  7/8 background\n"
            f" q  quit"
        )
        plt.render()

    plt.add_callback("LeftButtonPress", on_click)

    cam = _CAMERAS.get(args.view)
    if cam:
        plt.show(camera=cam)
    else:
        plt.show(viewup="y", zoom=1.2)


if __name__ == "__main__":
    main()

"""
3D visualization of the 12 rotations actually used in the TinyHAR/DeepConvLSTM
same-vertex (v2208) orientation sweep: each is a pure twist about a shared Z
axis (R_z(z_deg), z_deg = 0, 30, ..., 330), applied to an exemplary frame drawn
aligned with the world/inertial axes for clarity (not the real, tilted SMPL
surface frame -- the point here is to show the effect of the twist itself,
decoupled from wherever the vertex's normal happens to point on the body).

Because every one of these 12 rotations is about the same axis, the Z arrow
is identical across all of them -- drawn once -- while the X/Y arrows fan out
around the circle, colored by a cyclic hue keyed to z_deg (a genuinely
periodic variable, so a hue-cycle colormap is the right tool here, unlike for
ordinary categorical data). Placement A's own actual twist (z=300 deg) is
highlighted since it's the reference every angular distance in the sweep is
measured against.
"""

import colorsys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from PIL import Image
from pypdf import PdfReader, PdfWriter
from scipy.spatial.transform import Rotation

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
Z_DEGS = list(range(0, 360, 30))  # the 12 sweep points
A_Z_DEG = 300  # placement A's own actual twist, for reference

E1, E2, E3 = np.eye(3)[0], np.eye(3)[1], np.eye(3)[2]
AXIS_LEN = 1.0


def hue_color(z_deg: float) -> str:
    """Cyclic hue keyed directly to z_deg, since z_deg is itself a periodic
    (angular) variable -- 0 and 360 deg should, and do, map to the same hue."""
    h = (z_deg % 360) / 360.0
    r, g, b = colorsys.hls_to_rgb(h, 0.48, 0.85)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def arrow_trace(origin: np.ndarray, tip: np.ndarray, color: str, name: str,
                 width: int = 4, dash: str = "solid", hovertext: str | None = None) -> go.Scatter3d:
    return go.Scatter3d(
        x=[origin[0], tip[0]], y=[origin[1], tip[1]], z=[origin[2], tip[2]],
        mode="lines+markers",
        line=dict(color=color, width=width, dash=dash),
        marker=dict(size=[0, 5], color=color),
        name=name, legendgroup=name,
        hovertext=hovertext or name, hoverinfo="text",
        showlegend=False,
    )


def unit_circle_trace() -> go.Scatter3d:
    """Faint reference circle in the XY plane (Z=0): every X-arrow tip lies
    exactly on it, anchoring that all 12 rotations sweep within one plane."""
    t = np.linspace(0, 2 * np.pi, 200)
    return go.Scatter3d(
        x=np.cos(t) * AXIS_LEN, y=np.sin(t) * AXIS_LEN, z=np.zeros_like(t),
        mode="lines", line=dict(color="#c9c9c9", width=2),
        hoverinfo="skip", showlegend=False,
    )


def build_figure() -> go.Figure:
    origin = np.zeros(3)
    fig = go.Figure()

    fig.add_trace(unit_circle_trace())

    # Shared Z axis: identical for every rotation (it's the rotation axis
    # itself), so it's drawn exactly once, not once per frame.
    fig.add_trace(arrow_trace(
        origin, E3 * AXIS_LEN * 1.15, color="#2b2b2b", name="Z (shared axis, invariant)",
        width=7, hovertext="Z axis -- identical for all 12 rotations (the rotation axis itself)",
    ))
    fig.add_trace(go.Scatter3d(
        x=[E3[0] * AXIS_LEN * 1.25], y=[E3[1] * AXIS_LEN * 1.25], z=[E3[2] * AXIS_LEN * 1.25],
        mode="text", text=["Z (shared)"], textfont=dict(size=11, color="#2b2b2b"),
        showlegend=False,
    ))

    for z_deg in Z_DEGS:
        R = Rotation.from_euler("z", z_deg, degrees=True).as_matrix()
        x_axis, y_axis = R @ E1, R @ E2
        is_a = (z_deg == A_Z_DEG)
        color = "#111111" if is_a else hue_color(z_deg)
        width = 7 if is_a else 4
        label = "A (reference)" if is_a else f"z={z_deg}deg"

        # X = solid line, Y = dashed -- same hue per frame (so a frame's two
        # axes read as "belonging together"), but unambiguous even in a
        # static screenshot, unlike distinguishing them by width alone.
        fig.add_trace(arrow_trace(
            origin, x_axis * AXIS_LEN, color=color, name=f"X, {label}", width=width, dash="solid",
            hovertext=f"X axis, {label}",
        ))
        fig.add_trace(arrow_trace(
            origin, y_axis * AXIS_LEN, color=color, name=f"Y, {label}", width=width, dash="dash",
            hovertext=f"Y axis, {label}",
        ))
        if is_a:
            fig.add_trace(go.Scatter3d(
                x=[x_axis[0] * 1.15], y=[x_axis[1] * 1.15], z=[x_axis[2] * 1.15],
                mode="text", text=["A"], textfont=dict(size=12, color="#111111"),
                showlegend=False,
            ))

    fig.update_layout(
        template="plotly_white",
        width=760, height=380,
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            domain=dict(x=[0, 1], y=[0, 1]),
            xaxis=dict(title="x", range=[-1.3, 1.3], showticklabels=False),
            yaxis=dict(title="y", range=[-1.3, 1.3], showticklabels=False),
            # Data only occupies Z in roughly [0, 1.15] (the XY fan sits at
            # Z=0, the shared Z arrow points up) -- a symmetric [-1.3, 1.3]
            # range wastes half the box on empty negative-Z space. A tight,
            # asymmetric range plus a manual (non-cube) aspect ratio actually
            # flattens the rendered box to match the data instead of forcing
            # a cube regardless of content.
            zaxis=dict(title="z", range=[-0.15, 1.3], showticklabels=False),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=0.45),
            camera=dict(eye=dict(x=1.7, y=1.7, z=0.9)),
        ),
        showlegend=False,
    )
    return fig


def autocrop_png(png_path: Path, padding: int = 20) -> tuple[float, float, float, float]:
    """Crop away Plotly's built-in whitespace margins around the 3D scene
    (not controllable via layout/margin/domain settings for 3D exports), by
    detecting the bounding box of non-white content directly. Returns the
    crop box as fractions (left, top, right, bottom) of the original image,
    image-space (top=0 at the top), for reuse on the PDF export."""
    img = Image.open(png_path).convert("RGB")
    w, h = img.size
    gray = img.convert("L")
    mask = Image.eval(gray, lambda p: 0 if p > 245 else 255)
    bbox = mask.getbbox()
    if bbox is None:
        return (0.0, 0.0, 1.0, 1.0)

    left, top, right, bottom = bbox
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(w, right + padding)
    bottom = min(h, bottom + padding)

    img.crop((left, top, right, bottom)).save(png_path)
    return (left / w, top / h, right / w, bottom / h)


def crop_pdf_matching(pdf_path: Path, frac_box: tuple[float, float, float, float]) -> None:
    """Apply the same fractional crop box computed from the PNG to this
    (single-page) PDF's media/crop box, accounting for the image-vs-PDF
    vertical axis flip (image y grows downward, PDF y grows upward)."""
    left_f, top_f, right_f, bottom_f = frac_box
    reader = PdfReader(str(pdf_path))
    page = reader.pages[0]
    mb = page.mediabox
    x0, y0, x1, y1 = float(mb.left), float(mb.bottom), float(mb.right), float(mb.top)
    width, height = x1 - x0, y1 - y0

    new_x0, new_x1 = x0 + left_f * width, x0 + right_f * width
    new_y1, new_y0 = y1 - top_f * height, y1 - bottom_f * height

    for box in (page.mediabox, page.cropbox):
        box.lower_left = (new_x0, new_y0)
        box.upper_right = (new_x1, new_y1)

    writer = PdfWriter()
    writer.add_page(page)
    with open(pdf_path, "wb") as f:
        writer.write(f)


if __name__ == "__main__":
    fig = build_figure()
    plots_dir = EXPERIMENTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    html_path = plots_dir / "mismatch_rotation_frames_same_vertex.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved interactive HTML -> {html_path}")

    png_path = plots_dir / "mismatch_rotation_frames_same_vertex.png"
    fig.write_image(png_path, scale=2)
    frac_box = autocrop_png(png_path)
    print(f"Saved + auto-cropped PNG (single camera angle, preview only -- use the HTML to rotate) -> {png_path}")

    pdf_path = plots_dir / "mismatch_rotation_frames_same_vertex.pdf"
    # This is a WebGL-rendered 3D scene, so Kaleido embeds it as a raster
    # image inside the PDF container rather than true vector paths (unlike a
    # 2D Plotly chart) -- scale controls that embedded image's resolution,
    # same as for the PNG.
    fig.write_image(pdf_path, scale=3)
    crop_pdf_matching(pdf_path, frac_box)
    print(f"Saved + auto-cropped PDF (for LaTeX, same crop as the PNG) -> {pdf_path}")

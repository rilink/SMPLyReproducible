"""
Dose-response plot for a model's same-vertex (v2208, z-axis-twist-only)
orientation sweep: Experiment 0's baseline (angle=0, by definition -- same
placement, no mismatch) plus the 12-point sweep, sorted by angular distance
from placement A. Replaces the three near-identical per-model scripts this
was copy-pasted from (tinyhar/lightgbm/deepconvlstm_v2208) -- use --model.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from plot_rotation_frames_3d import hue_color, A_Z_DEG  # noqa: E402

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = EXPERIMENTS_DIR / "processed"


def _fold_stats(fold_metrics: dict) -> dict:
    accs = np.array([v["accuracy"] for v in fold_metrics.values()])
    f1s = np.array([v["macro_f1"] for v in fold_metrics.values()])
    return {
        "mean_accuracy": float(accs.mean()), "std_accuracy": float(accs.std()),
        "mean_macro_f1": float(f1s.mean()), "std_macro_f1": float(f1s.std()),
    }


def build_dataset(model: str) -> list[dict]:
    rows = []

    baseline = json.loads((PROCESSED_DIR / "experiment0_baseline_results.json").read_text())
    rows.append({
        "label": "baseline (A)", "angular_distance_deg": 0.0, "z_deg": A_Z_DEG,
        **_fold_stats(baseline[model]["fold_metrics"]),
    })

    sweep = json.loads((PROCESSED_DIR / f"orientation_sweep_{model}_v2208_A.json").read_text())
    for point in sweep:
        rows.append({
            "label": f"v2208 z={point['z_deg']}deg",
            "angular_distance_deg": point["angular_distance_deg"], "z_deg": point["z_deg"],
            "mean_accuracy": point["mean_accuracy"], "std_accuracy": point["std_accuracy"],
            "mean_macro_f1": point["mean_macro_f1"], "std_macro_f1": point["std_macro_f1"],
        })

    rows.sort(key=lambda r: r["angular_distance_deg"])
    return rows


def make_figure(rows: list[dict]) -> go.Figure:
    angles = [r["angular_distance_deg"] for r in rows]
    acc = [r["mean_accuracy"] for r in rows]
    acc_std = [r["std_accuracy"] for r in rows]
    f1 = [r["mean_macro_f1"] for r in rows]
    f1_std = [r["std_macro_f1"] for r in rows]
    labels = [r["label"] for r in rows]

    # Per-point marker color matches each rotation's hue in the 3D
    # rotation-frames plot (plot_rotation_frames_3d.py), so a point here can
    # be traced directly to its frame there. A's own frame keeps the same
    # black highlight in both plots.
    point_colors = [
        "#111111" if r["z_deg"] == A_Z_DEG else hue_color(r["z_deg"]) for r in rows
    ]
    line_acc, line_f1 = "#b7b7b7", "#8f8f8f"

    fig = go.Figure()

    for y, y_std, line_color, name, symbol in [
        (acc, acc_std, line_acc, "Accuracy", "circle"),
        (f1, f1_std, line_f1, "Macro-F1", "diamond"),
    ]:
        upper = [v + s for v, s in zip(y, y_std)]
        lower = [v - s for v, s in zip(y, y_std)]
        fig.add_trace(go.Scatter(
            x=angles + angles[::-1], y=upper + lower[::-1],
            fill="toself", fillcolor=line_color, opacity=0.10,
            line=dict(width=0), hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=angles, y=y, mode="lines+markers", name=name,
            line=dict(color=line_color, width=1.5, dash="dot"),
            marker=dict(size=9, color=point_colors, symbol=symbol,
                        line=dict(width=1, color="white")),
            customdata=labels,
            hovertemplate="%{customdata}<br>angle=%{x:.1f}deg<br>" + name + "=%{y:.3f}<extra></extra>",
        ))

    fig.add_hline(
        y=acc[0], line=dict(color="#898781", width=1, dash="dash"),
        annotation_text="same-placement baseline", annotation_position="top left",
        annotation_font=dict(size=9, color="#898781"),
    )

    fig.update_layout(
        template="plotly_white",
        width=380, height=300,
        margin=dict(l=50, r=15, t=15, b=45),
        font=dict(family="Arial, Helvetica, sans-serif", size=10),
        xaxis=dict(title="Angular distance from placement A (deg)", range=[-5, 182], dtick=30),
        yaxis=dict(title="Score", range=[0.2, 0.9]),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=9)),
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["tinyhar", "lightgbm", "deepconvlstm"])
    args = parser.parse_args()
    model = args.model
    tag = f"mismatch_dose_response_{model}_same_vertex"

    rows = build_dataset(model)

    csv_path = PROCESSED_DIR / f"{tag}.csv"
    with open(csv_path, "w") as f:
        f.write("label,angular_distance_deg,mean_accuracy,std_accuracy,mean_macro_f1,std_macro_f1\n")
        for r in rows:
            f.write(f"{r['label']},{r['angular_distance_deg']:.3f},{r['mean_accuracy']:.4f},"
                    f"{r['std_accuracy']:.4f},{r['mean_macro_f1']:.4f},{r['std_macro_f1']:.4f}\n")
    print(f"Saved consolidated CSV -> {csv_path}")

    json_path = PROCESSED_DIR / f"{tag}.json"
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"Saved consolidated JSON -> {json_path}")

    fig = make_figure(rows)
    plots_dir = EXPERIMENTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    html_path = plots_dir / f"{tag}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved interactive HTML -> {html_path}")

    pdf_path = plots_dir / f"{tag}.pdf"
    fig.write_image(pdf_path)
    print(f"Saved PDF (for LaTeX) -> {pdf_path}")

    svg_path = plots_dir / f"{tag}.svg"
    fig.write_image(svg_path)
    print(f"Saved SVG -> {svg_path}")

    png_path = plots_dir / f"{tag}.png"
    fig.write_image(png_path, scale=3)
    print(f"Saved PNG (preview) -> {png_path}")

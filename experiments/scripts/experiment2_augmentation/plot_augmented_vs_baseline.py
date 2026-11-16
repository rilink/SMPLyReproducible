"""
Compares an augmented-trained model (real A + rotation-jittered copies, see
augmented_training.py) against the original baseline model (real A only),
both evaluated at the same signed rotation offsets from placement A (e.g.
-60, -30, 0 (real, unrotated data), +30, +60 deg -- read directly from
whichever results file is loaded, not hardcoded, since different runs have
used different offset magnitudes (e.g. an earlier +-20/+-40 config).

Two orthogonal visual channels: color encodes which model was trained
(baseline vs. augmented), marker shape/line dash encodes which score
(accuracy vs. macro-F1) -- so all four series stay distinguishable without
overloading a single channel.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = EXPERIMENTS_DIR / "processed"

# Categorical colors for the two training conditions (fixed assignment, not
# cycled): baseline stays neutral/grey-blue since it's the reference model,
# augmented gets the accent color since it's the condition being tested.
COLOR_BASELINE = "#6b7a8f"
COLOR_AUGMENTED = "#c1440e"


def _load(model: str, tag: str) -> dict:
    path = PROCESSED_DIR / f"experiment2_augmented_{model}_{tag}.json"
    return json.loads(path.read_text())


def _offsets(results: dict) -> list[int]:
    return sorted((int(o) for o in results["augmented"]["0"].keys()), key=lambda o: o)


def _series(results: dict, condition: str, metric: str, offsets: list[int]) -> tuple[list[float], list[float]]:
    means, stds = [], []
    for offset in offsets:
        vals = [results[condition][str(fold)][str(offset)][metric] for fold in range(5)]
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
    return means, stds


def make_figure(data: dict, model: str) -> go.Figure:
    results = data["results"]
    offsets = _offsets(results)
    max_offset = max(abs(o) for o in offsets)
    fig = go.Figure()

    for condition, color, label in [
        ("baseline", COLOR_BASELINE, "Trained on original"),
        ("augmented", COLOR_AUGMENTED, "Trained on augmented"),
    ]:
        for metric, symbol, dash, metric_label in [
            ("accuracy", "circle", "solid", "Accuracy"),
            ("macro_f1", "diamond", "dash", "Macro-F1"),
        ]:
            means, stds = _series(results, condition, metric, offsets)
            upper = [m + s for m, s in zip(means, stds)]
            lower = [m - s for m, s in zip(means, stds)]
            fig.add_trace(go.Scatter(
                x=offsets + offsets[::-1], y=upper + lower[::-1],
                fill="toself", fillcolor=color, opacity=0.08,
                line=dict(width=0), hoverinfo="skip", showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=offsets, y=means, mode="lines+markers",
                name=f"{label}, {metric_label}",
                line=dict(color=color, width=2, dash=dash),
                marker=dict(size=9, symbol=symbol, color=color,
                            line=dict(width=1, color="white")),
                hovertemplate=f"{label}, {metric_label}<br>offset=%{{x:+d}}deg<br>score=%{{y:.3f}}<extra></extra>",
            ))

    fig.add_vline(x=0, line=dict(color="#c9c9c9", width=1, dash="dot"))
    fig.add_annotation(
        x=0, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
        text="original data", font=dict(size=9, color="#898781"),
    )

    fig.update_layout(
        template="plotly_white",
        width=460, height=340,
        margin=dict(l=50, r=15, t=35, b=45),
        font=dict(family="Arial, Helvetica, sans-serif", size=10),
        title=dict(text=model, font=dict(size=11)),
        xaxis=dict(title="Rotation offset from placement A (deg)",
                   tickmode="array", tickvals=offsets, range=[-max_offset - 8, max_offset + 8]),
        yaxis=dict(title="Score"),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0,
                    font=dict(size=9)),
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tinyhar", choices=["deepconvlstm", "cnnhar", "tinyhar"])
    parser.add_argument("--tag", default="30_60", help="Offset-config tag matching the results filename, e.g. 30_60 or 20_40")
    args = parser.parse_args()

    data = _load(args.model, args.tag)
    fig = make_figure(data, args.model)

    plots_dir = EXPERIMENTS_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)
    stem = f"augmentation_vs_baseline_{args.model}_{args.tag}"

    html_path = plots_dir / f"{stem}.html"
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"Saved interactive HTML -> {html_path}")

    pdf_path = plots_dir / f"{stem}.pdf"
    fig.write_image(pdf_path)
    print(f"Saved PDF (for LaTeX) -> {pdf_path}")

    png_path = plots_dir / f"{stem}.png"
    fig.write_image(png_path, scale=3)
    print(f"Saved PNG (preview) -> {png_path}")

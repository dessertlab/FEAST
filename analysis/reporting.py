"""Output organisation for the fusion experiment: CSVs, SVGs and PNGs.

Layout (per language and canonical level), rooted at ``results_dir``:

    <results_dir>/
      config.json                    run config + restriction statistics
      canonical_map.csv              raw CWE -> family audit table
      folds.csv                      sample_id -> fold
      calibration_reliability.csv    per (fold, tool, family) confusion + metrics
      fusion_predictions.csv         per (fold, strategy, family, row)
      fusion_metrics_per_family.csv  per (strategy, family) mean over folds + support
      fusion_metrics_overall.csv     per strategy: support-weighted / macro / median
      fusion_detection_overall.csv   per strategy: vuln/safe detection + attribution acc
      plots/<metric>.svg, .png       strategy comparison bar charts
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd

# Metrics charted by default (the support-weighted reductions from aggregation).
PLOT_METRICS = ("f2", "f1", "recall", "precision", "mcc")


def save_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_config(config: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    return path


# ── plots ─────────────────────────────────────────────────────────────────────

def _bar_svg(values: list[tuple[str, float | None]], title: str, width: int = 960) -> str:
    row_h, label_w = 28, 300
    plot_w = width - label_w - 90
    height = 54 + max(len(values), 1) * row_h
    finite = [v for _n, v in values if v is not None]
    lo, hi = min([0.0, *finite]) if finite else 0.0, max([1.0, *finite]) if finite else 1.0
    span = (hi - lo) or 1.0
    zero_x = label_w + int((0 - lo) / span * plot_w)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px}.t{font-size:16px;font-weight:700}'
        '.ax{stroke:#888;stroke-width:1}.bar{fill:#3568a8}.v{fill:#222}</style>',
        f'<text class="t" x="0" y="20">{escape(title)}</text>',
        f'<line class="ax" x1="{zero_x}" y1="34" x2="{zero_x}" y2="{height - 10}"/>',
    ]
    for i, (name, value) in enumerate(values):
        y = 42 + i * row_h
        parts.append(f'<text x="0" y="{y + 15}">{escape(str(name))}</text>')
        if value is None:
            parts.append(f'<text class="v" x="{label_w}" y="{y + 15}">n/a</text>')
            continue
        x = label_w + int((min(value, 0) - lo) / span * plot_w)
        end = label_w + int((max(value, 0) - lo) / span * plot_w)
        parts.append(f'<rect class="bar" x="{min(x, end)}" y="{y}" width="{max(abs(end - x), 2)}" height="19" rx="2"/>')
        parts.append(f'<text class="v" x="{label_w + plot_w + 10}" y="{y + 15}">{value:.3f}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def _save_png(values: list[tuple[str, float | None]], title: str, path: Path) -> None:
    """Horizontal bar chart via matplotlib (best-effort; skipped if unavailable)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    names = [n for n, _v in values]
    vals = [0.0 if v is None else v for _n, v in values]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(values) + 1.5))
    ax.barh(names, vals, color="#3568a8")
    ax.set_xlabel(title)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_strategy_plots(
    overall: pd.DataFrame,
    out_dir: Path,
    metrics: Iterable[str] = PLOT_METRICS,
    suffix: str = "_weighted",
) -> list[Path]:
    """One bar chart (SVG + PNG) per metric, comparing strategies on ``<metric><suffix>``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for metric in metrics:
        column = f"{metric}{suffix}"
        if column not in overall.columns:
            continue
        ordered = overall.sort_values(column, ascending=False, na_position="last")
        values = [(str(r["strategy"]), (None if pd.isna(r[column]) else float(r[column])))
                  for _i, r in ordered.iterrows()]
        title = f"{metric} (support-weighted)"
        svg_path = out_dir / f"{metric}.svg"
        svg_path.write_text(_bar_svg(values, title), encoding="utf-8")
        _save_png(values, title, out_dir / f"{metric}.png")
        written.append(svg_path)
    return written

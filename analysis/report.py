
from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_REPORT_METRICS = ("precision", "recall", "specificity", "f1", "f2", "mcc", "roc_auc", "pr_auc", "balanced_accuracy")


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def aggregate_by_strategy(metrics: pd.DataFrame, metric_columns: Iterable[str] = DEFAULT_REPORT_METRICS) -> pd.DataFrame:
    available = [metric for metric in metric_columns if metric in metrics.columns]
    if "strategy" not in metrics.columns:
        raise ValueError("metrics must contain a strategy column")
    return metrics.groupby("strategy", as_index=False)[available].mean(numeric_only=True)


def _bar_svg(values: list[tuple[str, float]], title: str, width: int = 900) -> str:
    row_h = 30
    label_w = 260
    plot_w = width - label_w - 80
    height = 54 + max(len(values), 1) * row_h
    finite = [value for _name, value in values if value is not None]
    lo = min([0.0, *finite]) if finite else 0.0
    hi = max([1.0, *finite]) if finite else 1.0
    span = hi - lo or 1.0
    zero_x = label_w + int((0 - lo) / span * plot_w)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:12px}.title{font-size:16px;font-weight:700}.axis{stroke:#888;stroke-width:1}.bar{fill:#3568a8}.value{fill:#222}</style>',
        f'<text class="title" x="0" y="20">{escape(title)}</text>',
        f'<line class="axis" x1="{zero_x}" y1="34" x2="{zero_x}" y2="{height - 10}"/>',
    ]
    for i, (name, value) in enumerate(values):
        y = 42 + i * row_h
        parts.append(f'<text x="0" y="{y + 16}">{escape(name)}</text>')
        if value is None:
            parts.append(f'<text class="value" x="{label_w}" y="{y + 16}">n/a</text>')
            continue
        x = label_w + int((min(value, 0) - lo) / span * plot_w)
        end = label_w + int((max(value, 0) - lo) / span * plot_w)
        bar_x = min(x, end)
        bar_w = max(abs(end - x), 2)
        parts.append(f'<rect class="bar" x="{bar_x}" y="{y}" width="{bar_w}" height="20" rx="2"/>')
        parts.append(f'<text class="value" x="{label_w + plot_w + 10}" y="{y + 16}">{value:.3f}</text>')
    parts.append('</svg>')
    return "\n".join(parts)


def save_fusion_report(
    metrics: pd.DataFrame,
    out_dir: str | Path = "data/results/plots",
    metric_columns: Iterable[str] = DEFAULT_REPORT_METRICS,
) -> dict[str, Path]:
    """Write lightweight HTML/SVG comparisons for fusion strategies."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_by_strategy(metrics, metric_columns)
    aggregate_path = out_dir / "strategy_metric_means.csv"
    aggregate.to_csv(aggregate_path, index=False)

    paths: dict[str, Path] = {"aggregate_csv": aggregate_path}
    svgs: list[str] = []
    for metric in [m for m in metric_columns if m in aggregate.columns]:
        values = []
        for row in aggregate.sort_values("strategy").to_dict("records"):
            values.append((str(row["strategy"]), _safe_float(row.get(metric))))
        svg = _bar_svg(values, metric)
        svg_path = out_dir / f"{metric}.svg"
        svg_path.write_text(svg, encoding="utf-8")
        paths[f"{metric}_svg"] = svg_path
        svgs.append(svg)

    html = "\n".join([
        '<!doctype html><html><head><meta charset="utf-8"><title>FEAST fusion report</title>',
        '<style>body{font-family:Arial,sans-serif;margin:24px;color:#222}section{margin:0 0 28px}h1{font-size:22px}svg{max-width:100%;height:auto}</style>',
        '</head><body><h1>FEAST fusion report</h1>',
        *[f'<section>{svg}</section>' for svg in svgs],
        '</body></html>',
    ])
    html_path = out_dir / "fusion_report.html"
    html_path.write_text(html, encoding="utf-8")
    paths["html"] = html_path
    return paths

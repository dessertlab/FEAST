"""Paired 95% confidence intervals for fusion-strategy mean differences.

The input is the per-family table produced by ``analysis.aggregation.mean_over_folds``:
one row per ``(strategy, family)`` with metrics already averaged across CV folds. For
each fusion strategy, this module compares the family-level metric values against two
fixed baselines (traditional K-of-N and OR), computes the paired mean difference and
writes a two-panel forest plot.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Sequence

import pandas as pd

from analysis.aggregation import REPORT_METRICS
from analysis.fusion.common import FUSER_ORDER, canonical_strategy_order, split_tau_strategy

LOWER_IS_BETTER = {"fpr", "fnr"}
DEFAULT_BASELINES = ("traditional", "or")
CI_LEVEL = 0.95

STATUS_IMPROVEMENT = "incremento_statistico"
STATUS_INCONCLUSIVE = "inconcludente"
STATUS_DEGRADATION = "degradazione_statistica"

STATUS_LABELS = {
    STATUS_IMPROVEMENT: "Incremento statistico",
    STATUS_INCONCLUSIVE: "Inconcludente",
    STATUS_DEGRADATION: "Degradazione statistica",
}

# Okabe-Ito palette: high-contrast and commonly used as colorblind-friendly.
STATUS_STYLES = {
    STATUS_IMPROVEMENT: {"color": "#0072B2", "marker": "^"},
    STATUS_INCONCLUSIVE: {"color": "#999999", "marker": "o"},
    STATUS_DEGRADATION: {"color": "#D55E00", "marker": "v"},
}

_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _t_critical_975(df: int) -> float:
    """Two-sided 95% t critical value, with a normal fallback for large df."""
    if df <= 0:
        return float("nan")
    if df in _T_975:
        return _T_975[df]
    if df <= 40:
        return 2.021
    if df <= 60:
        return 2.000
    if df <= 120:
        return 1.980
    return NormalDist().inv_cdf(0.975)


def _resolve_baseline(present: set[str], baseline: str) -> str:
    baseline = baseline.strip()
    if baseline in present:
        return baseline
    if baseline == "traditional":
        matches = sorted(s for s in present if s.startswith("traditional_"))
    elif baseline == "or":
        matches = sorted(s for s in present if s.startswith("or_1_of_"))
    else:
        matches = sorted(s for s in present if s.startswith(baseline))
    if not matches:
        raise ValueError(f"baseline strategy not found: {baseline}")
    if len(matches) > 1:
        raise ValueError(f"baseline {baseline!r} is ambiguous: {matches}")
    return matches[0]


def _fusion_strategies(present: Iterable[str]) -> list[str]:
    fusers = set(FUSER_ORDER)
    strategies: list[str] = []
    for strategy in present:
        base, _tau = split_tau_strategy(str(strategy))
        if base in fusers:
            strategies.append(str(strategy))
    return strategies


def _paired_difference(
    per_family: pd.DataFrame,
    strategy: str,
    baseline: str,
    metric: str,
) -> pd.Series:
    strategy_values = (
        per_family.loc[per_family["strategy"] == strategy, ["family", metric]]
        .rename(columns={metric: "strategy_value"})
    )
    baseline_values = (
        per_family.loc[per_family["strategy"] == baseline, ["family", metric]]
        .rename(columns={metric: "baseline_value"})
    )
    paired = strategy_values.merge(baseline_values, on="family", how="inner")
    strategy_metric = pd.to_numeric(paired["strategy_value"], errors="coerce")
    baseline_metric = pd.to_numeric(paired["baseline_value"], errors="coerce")
    return (strategy_metric - baseline_metric).dropna().reset_index(drop=True)


def _classify_difference(metric: str, ci_low: float, ci_high: float) -> str:
    if pd.isna(ci_low) or pd.isna(ci_high):
        return STATUS_INCONCLUSIVE
    lower_is_better = metric in LOWER_IS_BETTER
    if ci_low > 0:
        return STATUS_DEGRADATION if lower_is_better else STATUS_IMPROVEMENT
    if ci_high < 0:
        return STATUS_IMPROVEMENT if lower_is_better else STATUS_DEGRADATION
    return STATUS_INCONCLUSIVE


def mean_difference_ci(
    per_family: pd.DataFrame,
    metric: str = "f1",
    baselines: Sequence[str] = DEFAULT_BASELINES,
    tools: Sequence[str] = (),
) -> pd.DataFrame:
    """Compute paired 95% CIs for each fusion strategy against each baseline.

    ``difference`` is always ``strategy metric - baseline metric``. The ``status`` column
    accounts for metric direction, so lower-is-better metrics treat negative intervals as
    improvements.
    """
    required = {"strategy", "family", metric}
    missing = required - set(per_family.columns)
    if missing:
        raise ValueError(f"per_family is missing required columns: {sorted(missing)}")
    if metric not in REPORT_METRICS:
        raise ValueError(f"unsupported metric {metric!r}; expected one of {REPORT_METRICS}")

    present = set(per_family["strategy"].astype(str))
    resolved = [(name, _resolve_baseline(present, name)) for name in baselines]
    strategies = canonical_strategy_order(tools, _fusion_strategies(present))

    rows: list[dict] = []
    for baseline_name, baseline_strategy in resolved:
        for strategy in strategies:
            diffs = _paired_difference(per_family, strategy, baseline_strategy, metric)
            n = int(len(diffs))
            mean = float(diffs.mean()) if n else float("nan")
            if n <= 1:
                std = se = ci_low = ci_high = float("nan")
            else:
                std = float(diffs.std(ddof=1))
                se = std / sqrt(n)
                margin = _t_critical_975(n - 1) * se
                ci_low = mean - margin
                ci_high = mean + margin
            rows.append({
                "baseline": baseline_name,
                "baseline_strategy": baseline_strategy,
                "strategy": strategy,
                "metric": metric,
                "n_families": n,
                "mean_difference": mean,
                "std_difference": std,
                "standard_error": se,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_level": CI_LEVEL,
                "status": _classify_difference(metric, ci_low, ci_high),
            })
    return pd.DataFrame(rows)


def _strategy_label(strategy: str) -> str:
    base, tau = split_tau_strategy(strategy)
    label = base
    if base.startswith("weighted_fire_") and "_silence_" in base:
        fire, silence = base.removeprefix("weighted_fire_").split("_silence_", 1)
        label = f"weighted {fire}/{silence}"
    elif base.startswith("dst_") and "_fire_" in base and "_silence_" in base:
        rule, rest = base.removeprefix("dst_").split("_fire_", 1)
        fire, silence = rest.split("_silence_", 1)
        label = f"dst {rule} {fire}/{silence}"
    if tau is not None:
        label = f"{label} tau={tau:.1f}"
    return label


def save_mean_difference_ci_plot(
    intervals: pd.DataFrame,
    out_stem: Path,
    *,
    title: str | None = None,
) -> list[Path]:
    """Save the two-subplot CI forest plot as SVG and PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    required = {"baseline", "strategy", "metric", "mean_difference", "ci_low", "ci_high", "status"}
    missing = required - set(intervals.columns)
    if missing:
        raise ValueError(f"intervals is missing required columns: {sorted(missing)}")
    if intervals.empty:
        return []

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    baselines = list(dict.fromkeys(intervals["baseline"].astype(str)))
    strategies = list(dict.fromkeys(intervals["strategy"].astype(str)))
    labels = [_strategy_label(strategy) for strategy in strategies]
    metric = str(intervals["metric"].iloc[0])

    fig_h = max(4.8, 0.38 * len(strategies) + 2.2)
    fig_w = 15 if len(baselines) > 1 else 8
    fig, axes = plt.subplots(1, len(baselines), figsize=(fig_w, fig_h), sharey=True, squeeze=False)
    axes = axes[0]
    y_positions = list(range(len(strategies)))

    for ax, baseline in zip(axes, baselines, strict=False):
        frame = intervals[intervals["baseline"].astype(str) == baseline].set_index("strategy")
        baseline_title = (
            str(frame["baseline_strategy"].iloc[0])
            if "baseline_strategy" in frame.columns and not frame.empty
            else baseline
        )
        ax.axvline(0.0, color="#333333", linewidth=1.0, alpha=0.8)
        for y, strategy in zip(y_positions, strategies, strict=False):
            if strategy not in frame.index:
                continue
            row = frame.loc[strategy]
            mean = float(row["mean_difference"])
            ci_low = float(row["ci_low"])
            ci_high = float(row["ci_high"])
            status = str(row["status"])
            style = STATUS_STYLES.get(status, STATUS_STYLES[STATUS_INCONCLUSIVE])
            if pd.isna(mean) or pd.isna(ci_low) or pd.isna(ci_high):
                continue
            ax.errorbar(
                mean, y,
                xerr=[[mean - ci_low], [ci_high - mean]],
                fmt=style["marker"],
                color=style["color"],
                ecolor=style["color"],
                elinewidth=1.5,
                capsize=3,
                markersize=6,
            )
        ax.set_title(f"Confronto con {baseline_title}")
        ax.set_xlabel(f"Delta medio {metric}")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].set_yticks(y_positions, labels)
    axes[0].invert_yaxis()
    legend_handles = [
        mlines.Line2D(
            [], [], color=style["color"], marker=style["marker"], linestyle="None",
            markersize=7, label=STATUS_LABELS[status],
        )
        for status, style in STATUS_STYLES.items()
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle(title or f"IC 95% della differenza media ({metric})", y=0.995)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    svg_path = out_stem.with_suffix(".svg")
    png_path = out_stem.with_suffix(".png")
    fig.savefig(svg_path)
    fig.savefig(png_path, dpi=140)
    plt.close(fig)
    return [svg_path, png_path]


def save_mean_difference_ci_report(
    per_family: pd.DataFrame,
    out_dir: Path,
    *,
    metric: str = "f1",
    baselines: Sequence[str] = DEFAULT_BASELINES,
    tools: Sequence[str] = (),
    stem: str | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Compute intervals, write the CSV, and save the SVG/PNG two-panel plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    intervals = mean_difference_ci(per_family, metric=metric, baselines=baselines, tools=tools)
    name = stem or f"mean_difference_ci_{metric}"
    csv_path = out_dir / f"{name}.csv"
    intervals.to_csv(csv_path, index=False)
    plot_paths = save_mean_difference_ci_plot(intervals, out_dir / name)
    return intervals, [csv_path, *plot_paths]

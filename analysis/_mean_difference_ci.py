"""Paired 95% confidence intervals for fusion-strategy mean differences.

Two pairing modes are supported:

* **fold** (default, recommended): for each strategy the K-fold cross-validation gives K
  independent support-weighted aggregate values (one per fold).  The CI is on the K
  paired differences ``strategy_fold_k - baseline_fold_k``.  This aligns with the
  "overall (support-weighted across families)" headline table: the metric being compared
  is the same support-weighted quantity, and the fold is the natural replication unit.
  Requires ``per_family_per_fold`` (the raw fold-level data saved as
  ``fusion_metrics_per_family_per_fold.csv``).

* **family** (legacy): pairs across CWE families using fold-averaged metrics.  N equals
  the number of families; each observation is ``metric(strategy, family) -
  metric(baseline, family)`` with no support weighting applied to the pairing.  This
  treats rare and common families as equally informative, which disagrees with the
  support-weighted headline table.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Sequence

import pandas as pd

from analysis.aggregation import REPORT_METRICS, SUPPORT_COLUMN
from analysis.fusion.common import FUSER_ORDER, canonical_strategy_order, split_tau_strategy

LOWER_IS_BETTER = {"fpr", "fnr"}
DEFAULT_BASELINES = ("traditional",)
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
        if len(matches) > 1:
            matches_2 = [m for m in matches if m.startswith("traditional_2_of_")]
            if matches_2:
                return matches_2[0]
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


def _fold_weighted_series(
    per_family_per_fold: pd.DataFrame,
    strategy: str,
    metric: str,
) -> pd.Series:
    """Support-weighted metric per fold for one strategy (indexed by fold value)."""
    sub = per_family_per_fold[per_family_per_fold["strategy"] == strategy]
    result: dict = {}
    for fold, group in sub.groupby("fold"):
        values = pd.to_numeric(group[metric], errors="coerce")
        weights = pd.to_numeric(group[SUPPORT_COLUMN], errors="coerce")
        mask = values.notna() & weights.notna() & (weights > 0)
        total = float(weights[mask].sum())
        result[fold] = float((values[mask] * weights[mask]).sum() / total) if total > 0 else float("nan")
    return pd.Series(result)


def _paired_difference_folds(
    per_family_per_fold: pd.DataFrame,
    strategy: str,
    baseline: str,
    metric: str,
) -> pd.Series:
    """Fold-level paired differences: support-weighted(strategy) - support-weighted(baseline)."""
    strat = _fold_weighted_series(per_family_per_fold, strategy, metric)
    base = _fold_weighted_series(per_family_per_fold, baseline, metric)
    common = strat.index.intersection(base.index)
    return (strat[common] - base[common]).dropna().reset_index(drop=True)


def _best_tau_per_strategy(
    per_family: pd.DataFrame,
    strategies: list[str],
    metric: str,
) -> list[str]:
    """For each base strategy keep only the tau variant with the best mean metric.

    Strategies without a tau suffix are passed through unchanged.  When all
    variants of a base strategy have NaN for the metric the first variant is
    kept as a fallback so the base strategy still appears in the plot.
    """
    ascending = metric in LOWER_IS_BETTER
    groups: dict[str, list[str]] = {}
    for s in strategies:
        base, _tau = split_tau_strategy(s)
        groups.setdefault(base, []).append(s)

    selected: list[str] = []
    for _base, variants in groups.items():
        if len(variants) == 1:
            selected.append(variants[0])
            continue
        scores: dict[str, float] = {}
        for v in variants:
            vals = pd.to_numeric(
                per_family.loc[per_family["strategy"] == v, metric],
                errors="coerce",
            )
            scores[v] = float(vals.mean()) if vals.notna().any() else float("nan")
        valid = {v: s for v, s in scores.items() if not pd.isna(s)}
        if not valid:
            selected.append(variants[0])
        else:
            best = min(valid, key=lambda v: valid[v] if ascending else -valid[v])
            selected.append(best)
    return selected


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
    best_tau_only: bool = True,
    per_family_per_fold: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute paired 95% CIs for each fusion strategy against each baseline.

    ``difference`` is always ``strategy metric - baseline metric``. The ``status`` column
    accounts for metric direction, so lower-is-better metrics treat negative intervals as
    improvements.

    ``per_family_per_fold``: when provided, the CI is computed on K fold-level paired
    differences of the support-weighted metric (N = K folds).  This matches the
    "overall (support-weighted across families)" headline table.  When None, falls back
    to pairing across CWE families using the fold-averaged ``per_family`` table (N =
    number of families) — legacy behaviour.

    ``best_tau_only``: when True (default) each base strategy contributes only its
    best-mean-metric tau variant.  Tau selection always uses ``per_family`` (fold-averaged
    per-family values) regardless of the pairing mode.
    """
    required = {"strategy", "family", metric}
    missing = required - set(per_family.columns)
    if missing:
        raise ValueError(f"per_family is missing required columns: {sorted(missing)}")
    if metric not in REPORT_METRICS:
        raise ValueError(f"unsupported metric {metric!r}; expected one of {REPORT_METRICS}")

    present = set(per_family["strategy"].astype(str))
    if baselines == DEFAULT_BASELINES:
        trad_strategies = sorted(s for s in present if s.startswith("traditional_"))
        if len(trad_strategies) == 1:
            resolved = [("traditional", trad_strategies[0])]
        elif len(trad_strategies) > 1:
            resolved = []
            for s in trad_strategies:
                parts = s.split("_")
                if len(parts) >= 4 and parts[0] == "traditional" and parts[2] == "of":
                    resolved.append((f"traditional_{parts[1]}", s))
                else:
                    resolved.append((s, s))
        else:
            resolved = [(name, _resolve_baseline(present, name)) for name in baselines]
    else:
        resolved = [(name, _resolve_baseline(present, name)) for name in baselines]
    all_fusion = canonical_strategy_order(tools, _fusion_strategies(present))
    strategies = (
        _best_tau_per_strategy(per_family, all_fusion, metric) if best_tau_only else all_fusion
    )
    strategies = canonical_strategy_order(tools, strategies)

    rows: list[dict] = []
    for baseline_name, baseline_strategy in resolved:
        for strategy in strategies:
            diffs = (
                _paired_difference_folds(per_family_per_fold, strategy, baseline_strategy, metric)
                if per_family_per_fold is not None
                else _paired_difference(per_family, strategy, baseline_strategy, metric)
            )
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
            pairing = "fold" if per_family_per_fold is not None else "family"
            rows.append({
                "baseline": baseline_name,
                "baseline_strategy": baseline_strategy,
                "strategy": strategy,
                "metric": metric,
                "pairing": pairing,
                "n_pairs": n,
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
    """Save a paper-style CI forest plot as SVG and PNG."""
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
    unique_baselines = sorted(intervals["baseline"].astype(str).unique())

    if len(unique_baselines) <= 1:
        preferred = unique_baselines[0] if unique_baselines else "traditional"
        frame = intervals[intervals["baseline"].astype(str) == preferred].copy()
        if frame.empty:
            return []

        strategies = frame["strategy"].astype(str).tolist()
        labels = [_strategy_label(strategy) for strategy in strategies]
        metric = str(frame["metric"].iloc[0])
        baseline_title = (
            str(frame["baseline_strategy"].iloc[0])
            if "baseline_strategy" in frame.columns and not frame.empty
            else preferred
        )

        y_positions = list(range(len(strategies)))
        fig_h = max(4.4, 0.34 * len(strategies) + 1.8)
        fig_w = 8.2
        rc = {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            for y in y_positions:
                if y % 2:
                    ax.axhspan(y - 0.5, y + 0.5, color="#f7f7f7", zorder=0)
            ax.axvline(0.0, color="#222222", linewidth=0.9, linestyle="--", zorder=1)

            for y, (_idx, row) in zip(y_positions, frame.iterrows(), strict=False):
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
                    elinewidth=1.1,
                    capsize=2.5,
                    capthick=1.0,
                    markersize=5.2,
                    markeredgewidth=0.8,
                    markeredgecolor="#222222",
                    zorder=3,
                )

            ax.set_yticks(y_positions, labels)
            ax.invert_yaxis()
            ax.set_xlabel(f"Mean difference in {metric} (strategy - {baseline_title})")
            ax.set_ylabel("")
            ax.set_title(title or f"95% CI of paired mean differences vs {baseline_title}", fontsize=11, pad=10)
            ax.grid(axis="x", linestyle=":", linewidth=0.6, color="#bdbdbd", alpha=0.8)
            ax.tick_params(axis="both", labelsize=8.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

            legend_handles = [
                mlines.Line2D(
                    [], [], color=style["color"], marker=style["marker"], linestyle="None",
                    markeredgecolor="#222222", markeredgewidth=0.8, markersize=6,
                    label=STATUS_LABELS[status],
                )
                for status, style in STATUS_STYLES.items()
            ]
            ax.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=3,
                frameon=False,
                fontsize=8.5,
                handletextpad=0.4,
                columnspacing=1.2,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 1))

            svg_path = out_stem.with_suffix(".svg")
            png_path = out_stem.with_suffix(".png")
            fig.savefig(svg_path, bbox_inches="tight")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        return [svg_path, png_path]
    else:
        baseline_1, baseline_2 = unique_baselines[:2]
        frame_1 = intervals[intervals["baseline"].astype(str) == baseline_1].copy()
        frame_2 = intervals[intervals["baseline"].astype(str) == baseline_2].copy()

        if frame_1.empty or frame_2.empty:
            return []

        strategies = frame_1["strategy"].astype(str).tolist()
        labels = [_strategy_label(strategy) for strategy in strategies]
        metric = str(frame_1["metric"].iloc[0])
        baseline_title_1 = str(frame_1["baseline_strategy"].iloc[0]) if "baseline_strategy" in frame_1.columns else baseline_1
        baseline_title_2 = str(frame_2["baseline_strategy"].iloc[0]) if "baseline_strategy" in frame_2.columns else baseline_2

        # Align frame_2 to the same strategies order as frame_1
        frame_2 = frame_2.set_index("strategy").reindex(strategies).reset_index()

        y_positions = list(range(len(strategies)))
        fig_h = max(4.4, 0.34 * len(strategies) + 1.8)
        fig_w = 12.0
        rc = {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "text.color": "#222222",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
        with plt.rc_context(rc):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True)

            # Left subplot
            for y in y_positions:
                if y % 2:
                    ax1.axhspan(y - 0.5, y + 0.5, color="#f7f7f7", zorder=0)
            ax1.axvline(0.0, color="#222222", linewidth=0.9, linestyle="--", zorder=1)

            for y, (_idx, row) in zip(y_positions, frame_1.iterrows(), strict=False):
                mean = float(row["mean_difference"])
                ci_low = float(row["ci_low"])
                ci_high = float(row["ci_high"])
                status = str(row["status"])
                style = STATUS_STYLES.get(status, STATUS_STYLES[STATUS_INCONCLUSIVE])
                if pd.isna(mean) or pd.isna(ci_low) or pd.isna(ci_high):
                    continue
                ax1.errorbar(
                    mean, y,
                    xerr=[[mean - ci_low], [ci_high - mean]],
                    fmt=style["marker"],
                    color=style["color"],
                    ecolor=style["color"],
                    elinewidth=1.1,
                    capsize=2.5,
                    capthick=1.0,
                    markersize=5.2,
                    markeredgewidth=0.8,
                    markeredgecolor="#222222",
                    zorder=3,
                )

            ax1.set_yticks(y_positions, labels)
            ax1.invert_yaxis()
            ax1.set_xlabel(f"Mean difference in {metric} (strategy - {baseline_title_1})")
            ax1.set_title(f"vs {baseline_title_1}", fontsize=11, pad=10)
            ax1.grid(axis="x", linestyle=":", linewidth=0.6, color="#bdbdbd", alpha=0.8)
            ax1.tick_params(axis="both", labelsize=8.5)
            for spine in ("top", "right"):
                ax1.spines[spine].set_visible(False)

            # Right subplot
            for y in y_positions:
                if y % 2:
                    ax2.axhspan(y - 0.5, y + 0.5, color="#f7f7f7", zorder=0)
            ax2.axvline(0.0, color="#222222", linewidth=0.9, linestyle="--", zorder=1)

            for y, (_idx, row) in zip(y_positions, frame_2.iterrows(), strict=False):
                mean = float(row["mean_difference"])
                ci_low = float(row["ci_low"])
                ci_high = float(row["ci_high"])
                status = str(row["status"])
                style = STATUS_STYLES.get(status, STATUS_STYLES[STATUS_INCONCLUSIVE])
                if pd.isna(mean) or pd.isna(ci_low) or pd.isna(ci_high):
                    continue
                ax2.errorbar(
                    mean, y,
                    xerr=[[mean - ci_low], [ci_high - mean]],
                    fmt=style["marker"],
                    color=style["color"],
                    ecolor=style["color"],
                    elinewidth=1.1,
                    capsize=2.5,
                    capthick=1.0,
                    markersize=5.2,
                    markeredgewidth=0.8,
                    markeredgecolor="#222222",
                    zorder=3,
                )

            ax2.set_xlabel(f"Mean difference in {metric} (strategy - {baseline_title_2})")
            ax2.set_title(f"vs {baseline_title_2}", fontsize=11, pad=10)
            ax2.grid(axis="x", linestyle=":", linewidth=0.6, color="#bdbdbd", alpha=0.8)
            ax2.tick_params(axis="both", labelsize=8.5)
            for spine in ("top", "right"):
                ax2.spines[spine].set_visible(False)

            # Title & Legend for overall figure
            fig.suptitle(title or f"95% CI of paired mean differences vs Baselines", fontsize=12, y=0.98)
            legend_handles = [
                mlines.Line2D(
                    [], [], color=style["color"], marker=style["marker"], linestyle="None",
                    markeredgecolor="#222222", markeredgewidth=0.8, markersize=6,
                    label=STATUS_LABELS[status],
                )
                for status, style in STATUS_STYLES.items()
            ]
            fig.legend(
                handles=legend_handles,
                loc="lower center",
                bbox_to_anchor=(0.5, -0.08),
                ncol=3,
                frameon=False,
                fontsize=8.5,
                handletextpad=0.4,
                columnspacing=1.2,
            )
            fig.tight_layout(rect=(0, 0.04, 1, 0.95))

            svg_path = out_stem.with_suffix(".svg")
            png_path = out_stem.with_suffix(".png")
            fig.savefig(svg_path, bbox_inches="tight")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
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
    best_tau_only: bool = True,
    per_family_per_fold: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[Path]]:
    """Compute intervals, write the CSV, and save the SVG/PNG forest plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    intervals = mean_difference_ci(
        per_family, metric=metric, baselines=baselines, tools=tools,
        best_tau_only=best_tau_only, per_family_per_fold=per_family_per_fold,
    )
    name = stem or f"mean_difference_ci_{metric}"
    csv_path = out_dir / f"{name}.csv"
    intervals.to_csv(csv_path, index=False)
    plot_paths = save_mean_difference_ci_plot(intervals, out_dir / name)
    return intervals, [csv_path, *plot_paths]

"""Wilcoxon signed-rank test and Hodges-Lehmann confidence intervals for fusion-strategy differences.

This module replaces the parametric paired t-test (see ``_mean_difference_ci.py`` for the
staging/legacy version) with the non-parametric **Wilcoxon signed-rank test**.

Rationale for switching:
  The default pairing mode uses K=5 fold-level differences.  With N=5, the paired t-test
  relies heavily on the normality assumption, which cannot be verified and is frequently
  violated for F1/MCC differences.  The Wilcoxon signed-rank test is distribution-free and
  more appropriate for small samples; it also preserves statistical validity when the
  underlying distributions are skewed or heavy-tailed.

Statistical procedure:
  For each (strategy, baseline) pair, a vector of N paired differences d_i is computed.
  - **Test statistic**: scipy ``wilcoxon(d, alternative='two-sided', method='auto')``.
    With N<=25 the exact distribution is used; otherwise the normal approximation applies.
  - **Point estimate**: Hodges-Lehmann estimator — median of all N(N+1)/2 Walsh averages
    (d_i + d_j)/2, i<=j.  This is the natural point estimate dual to the Wilcoxon test.
  - **95% CI**: achieved-level CI via the Walsh-average approach (scipy>=1.11 exposes
    ``wilcoxon`` confidence_level).  For older scipy this is approximated by
    ±1.96*(IQR / 1.349) / sqrt(N) (normal-approximation on the ranks).
  - **Status**: identical classification logic as the t-test version: if the entire CI
    lies above 0 the strategy is classified as a statistically significant improvement;
    if entirely below 0, a degradation; otherwise inconclusive.

Two pairing modes are supported:

* **fold** (default, recommended): K=5 fold-level support-weighted aggregate values.
  Requires ``per_family_per_fold``.

* **family** (legacy): N = number of CWE families, fold-averaged metrics, no support
  weighting.
"""

from __future__ import annotations

from math import sqrt
import re
from pathlib import Path
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
    STATUS_IMPROVEMENT: "Statistical Improvement",
    STATUS_INCONCLUSIVE: "Inconclusive",
    STATUS_DEGRADATION: "Statistical Degradation",
}

# Okabe-Ito palette: high-contrast and commonly used as colorblind-friendly.
STATUS_STYLES = {
    STATUS_IMPROVEMENT: {"color": "#0072B2", "marker": "^"},
    STATUS_INCONCLUSIVE: {"color": "#999999", "marker": "o"},
    STATUS_DEGRADATION: {"color": "#D55E00", "marker": "v"},
}


def _wilcoxon_ci(
    diffs: "pd.Series",
    confidence_level: float = 0.95,
) -> tuple[float, float, float, float]:
    """Hodges-Lehmann point estimate and Walsh-average CI via Wilcoxon signed-rank test.

    Returns (p_value, hl_estimate, ci_low, ci_high).
    For N <= 1 all values are NaN.
    For N == 2 the Wilcoxon test cannot be computed (degenerate); falls back to
    midpoint ± half-range as a trivial interval.
    """
    import numpy as np
    from scipy.stats import wilcoxon

    d = diffs.dropna().to_numpy(dtype=float)
    n = len(d)
    if n <= 1:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Hodges-Lehmann estimator: median of Walsh averages (d_i + d_j)/2, i<=j
    idx = np.triu_indices(n)
    walsh = (d[idx[0]] + d[idx[1]]) / 2.0
    hl = float(np.median(walsh))

    if n == 2:
        # Wilcoxon requires at least 2 non-zero differences but with n=2 it is
        # degenerate (only 1 possible rank permutation per sign). Return a
        # trivial interval based on the observed range.
        ci_low = float(walsh.min())
        ci_high = float(walsh.max())
        return float("nan"), hl, ci_low, ci_high

    # scipy >= 1.11: wilcoxon accepts confidence_level and returns ci
    try:
        result = wilcoxon(d, alternative="two-sided", method="auto",
                          confidence_level=confidence_level)
        p_value = float(result.pvalue)
        ci = result.confidence_interval
        ci_low = float(ci.low)
        ci_high = float(ci.high)
    except TypeError:
        # Older scipy: compute p-value only; approximate CI via normal approx on Walsh averages
        result = wilcoxon(d, alternative="two-sided", method="auto")
        p_value = float(result.pvalue)
        walsh_std = float(walsh.std(ddof=1))
        margin = 1.96 * walsh_std / sqrt(len(walsh))
        ci_low = hl - margin
        ci_high = hl + margin

    return p_value, hl, ci_low, ci_high


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
) -> tuple["pd.Series", int, int]:
    """Return (diffs, n_coverage_gain, n_coverage_loss).

    ``n_coverage_gain`` counts families where the baseline is silent (F1=NaN)
    but the strategy detects something (F1>0).
    ``n_coverage_loss`` counts families where the strategy is silent (F1=NaN)
    but the baseline detects something (F1>0).

    The baseline NaN is treated as F1=0 (standard zero-division convention:
    when a deterministic classifier makes zero positive predictions its F1 is
    exactly 0).
    """
    strategy_values = (
        per_family.loc[per_family["strategy"] == strategy, ["family", metric]]
        .rename(columns={metric: "strategy_value"})
    )
    baseline_values = (
        per_family.loc[per_family["strategy"] == baseline, ["family", metric]]
        .rename(columns={metric: "baseline_value"})
    )
    paired = strategy_values.merge(baseline_values, on="family", how="inner")
    strategy_metric_raw = pd.to_numeric(paired["strategy_value"], errors="coerce")
    baseline_metric_raw = pd.to_numeric(paired["baseline_value"], errors="coerce")
    baseline_nan = baseline_metric_raw.isna()
    strategy_nan = strategy_metric_raw.isna()
    # F1=NaN means zero positive predictions (recall=0) for both deterministic baselines
    # and ML strategies.  Apply fillna(0.0) symmetrically so that coverage-loss pairs
    # (strategy silent, baseline detects) are included as negative differences rather
    # than silently dropped.  Pairs where BOTH are NaN carry no information and are
    # excluded explicitly.
    strategy_metric = strategy_metric_raw.fillna(0.0)
    baseline_metric = baseline_metric_raw.fillna(0.0)
    n_coverage_gain = int((baseline_nan & strategy_metric_raw.notna() & (strategy_metric_raw > 0)).sum())
    n_coverage_loss = int((strategy_nan & ~baseline_nan & (baseline_metric_raw > 0)).sum())
    both_nan = baseline_nan & strategy_nan
    diffs = (strategy_metric - baseline_metric)[~both_nan].reset_index(drop=True)
    return diffs, n_coverage_gain, n_coverage_loss


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
            # When multiple traditional_* variants exist, always prefer the
            # 2-of-N variant as the sole comparison baseline and discard the
            # rest (e.g. traditional_3_of_6 is never used).
            matches_2 = [s for s in trad_strategies if s.startswith("traditional_2_of_")]
            if matches_2:
                resolved = [("traditional_2", matches_2[0])]
            else:
                # No 2-of-N available: fall back to all variants as before.
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
            if per_family_per_fold is not None:
                diffs = _paired_difference_folds(
                    per_family_per_fold, strategy, baseline_strategy, metric
                )
                n_coverage_gain = 0
                n_coverage_loss = 0
            else:
                diffs, n_coverage_gain, n_coverage_loss = _paired_difference(
                    per_family, strategy, baseline_strategy, metric
                )
            n = int(len(diffs))
            mean = float(diffs.mean()) if n else float("nan")
            p_value, hl_estimate, ci_low, ci_high = _wilcoxon_ci(diffs, confidence_level=CI_LEVEL)
            pairing = "fold" if per_family_per_fold is not None else "family"
            rows.append({
                "baseline": baseline_name,
                "baseline_strategy": baseline_strategy,
                "strategy": strategy,
                "metric": metric,
                "pairing": pairing,
                "n_pairs": n,
                "mean_difference": mean,
                "hodges_lehmann": hl_estimate,
                "p_value": p_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_level": CI_LEVEL,
                "status": _classify_difference(metric, ci_low, ci_high),
                "n_coverage_gain": n_coverage_gain,
                "n_coverage_loss": n_coverage_loss,
            })
    return pd.DataFrame(rows)


def _strategy_label(strategy: str) -> str:
    base, tau = split_tau_strategy(strategy)
    if base.startswith("weighted_fire_") and "_silence_" in base:
        label = "Weighted Voting"
    elif base.startswith("dst_") and "_fire_" in base and "_silence_" in base:
        rule, _rest = base.removeprefix("dst_").split("_fire_", 1)
        rule_label = rule.title() if rule.lower() not in ("pcr6",) else "PCR6"
        label = f"DST {rule_label}"
    else:
        mapping = {
            "logistic_regression": "Logistic Regression",
            "decision_tree": "Decision Tree",
            "random_forest": "Random Forest",
            "gradient_boosting": "Gradient Boosting",
            "naive_bayes": "Naive Bayes",
            "logistic_interactions": "Logistic Interactions",
            "bks": "BKS",
        }
        # Pattern-based labels for N-of-M voting strategies; handles any tool count.
        m_or = re.match(r"or_(\d+)_of_(\d+)$", base)
        m_trad = re.match(r"traditional_(\d+)_of_(\d+)$", base)
        if m_or:
            label = f"OR {m_or.group(1)}-of-{m_or.group(2)}"
        elif m_trad:
            label = f"Traditional {m_trad.group(1)}-of-{m_trad.group(2)}"
        else:
            label = mapping.get(base, base.replace("_", " ").title())
            for acr in ["Dst", "Pcr6", "Bks", "Or"]:
                if acr in label:
                    label = label.replace(acr, acr.upper())
    if tau is not None:
        label = f"{label}, $\\tau$={tau:.1f}"
    return label


def _format_p_value(p_val: float) -> str:
    if p_val >= 0.001:
        return f"p={p_val:.3f}"
    s = f"{p_val:.1e}"
    s = s.replace("e-0", "e-").replace("e+0", "e+")
    return f"p={s}"


def _annotate_pvalue(
    ax: "matplotlib.axes.Axes",
    hl: float,
    y: float,
    row: "pd.Series",
) -> None:
    """Draw the p-value (+ optional Δc coverage indicator) below a CI point.

    For degenerate tests (p=NaN, e.g. n<=2), draws ``p=*`` when the coverage
    delta is non-zero, so that meaningful coverage information is not lost.
    """
    gain = int(row.get("n_coverage_gain", 0) or 0)
    loss = int(row.get("n_coverage_loss", 0) or 0)
    delta_c = gain - loss

    p_val_raw = row.get("p_value") if "p_value" in row else None
    p_is_nan = p_val_raw is None or pd.isna(p_val_raw)

    if p_is_nan:
        if delta_c == 0:
            return  # nothing informative to show
        p_str = "p=*"
        weight = "normal"
    else:
        p_val = float(p_val_raw)
        p_str = _format_p_value(p_val)
        weight = "bold" if p_val < 0.05 else "normal"

    if delta_c > 0:
        p_str += f" (+{delta_c})"
    elif delta_c < 0:
        p_str += f" ({delta_c})"

    ax.text(
        hl, y - 0.15, p_str,
        ha="center", va="bottom",
        fontsize=16, color="#000000",
        fontweight=weight,
        zorder=4,
    )


def save_mean_difference_ci_plot(
    intervals: pd.DataFrame,
    out_stem: Path,
    *,
    show_pvalues: bool = False,
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


        y_positions = list(range(len(strategies)))
        fig_h = max(4.4, 0.34 * len(strategies) + 1.8)
        fig_w = 8.2
        rc = {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.edgecolor": "#000000",
            "axes.linewidth": 0.8,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "text.color": "#000000",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "font.size": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            for y in y_positions:
                if y % 2:
                    ax.axhspan(y - 0.5, y + 0.5, color="#f7f7f7", zorder=0)
            ax.axvline(0.0, color="#000000", linewidth=0.9, linestyle="--", zorder=1)

            for y, (_idx, row) in zip(y_positions, frame.iterrows(), strict=False):
                hl = float(row["hodges_lehmann"])
                ci_low = float(row["ci_low"])
                ci_high = float(row["ci_high"])
                status = str(row["status"])
                style = STATUS_STYLES.get(status, STATUS_STYLES[STATUS_INCONCLUSIVE])
                if pd.isna(hl) or pd.isna(ci_low) or pd.isna(ci_high):
                    continue
                ax.errorbar(
                    hl, y,
                    xerr=[[hl - ci_low], [ci_high - hl]],
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
                if show_pvalues:
                    _annotate_pvalue(ax, hl, y, row)

            ax.set_yticks(y_positions, labels)
            ax.invert_yaxis()
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title("")
            ax.grid(axis="x", linestyle=":", linewidth=0.6, color="#bdbdbd", alpha=0.8)
            ax.tick_params(axis="both", labelsize=14)
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
                fontsize=14,
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
            "font.size": 14,
            "axes.labelsize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
        }
        with plt.rc_context(rc):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_w, fig_h), sharey=True)

            # Left subplot
            for y in y_positions:
                if y % 2:
                    ax1.axhspan(y - 0.5, y + 0.5, color="#f7f7f7", zorder=0)
            ax1.axvline(0.0, color="#222222", linewidth=0.9, linestyle="--", zorder=1)

            for y, (_idx, row) in zip(y_positions, frame_1.iterrows(), strict=False):
                hl = float(row["hodges_lehmann"])
                ci_low = float(row["ci_low"])
                ci_high = float(row["ci_high"])
                status = str(row["status"])
                style = STATUS_STYLES.get(status, STATUS_STYLES[STATUS_INCONCLUSIVE])
                if pd.isna(hl) or pd.isna(ci_low) or pd.isna(ci_high):
                    continue
                ax1.errorbar(
                    hl, y,
                    xerr=[[hl - ci_low], [ci_high - hl]],
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
                if show_pvalues:
                    _annotate_pvalue(ax1, hl, y, row)

            ax1.set_yticks(y_positions, labels)
            ax1.invert_yaxis()
            ax1.set_xlabel("")
            ax1.set_title("")
            ax1.grid(axis="x", linestyle=":", linewidth=0.6, color="#bdbdbd", alpha=0.8)
            ax1.tick_params(axis="both", labelsize=14)
            for spine in ("top", "right"):
                ax1.spines[spine].set_visible(False)

            # Right subplot
            for y in y_positions:
                if y % 2:
                    ax2.axhspan(y - 0.5, y + 0.5, color="#f7f7f7", zorder=0)
            ax2.axvline(0.0, color="#222222", linewidth=0.9, linestyle="--", zorder=1)

            for y, (_idx, row) in zip(y_positions, frame_2.iterrows(), strict=False):
                hl = float(row["hodges_lehmann"])
                ci_low = float(row["ci_low"])
                ci_high = float(row["ci_high"])
                status = str(row["status"])
                style = STATUS_STYLES.get(status, STATUS_STYLES[STATUS_INCONCLUSIVE])
                if pd.isna(hl) or pd.isna(ci_low) or pd.isna(ci_high):
                    continue
                ax2.errorbar(
                    hl, y,
                    xerr=[[hl - ci_low], [ci_high - hl]],
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
                if show_pvalues:
                    _annotate_pvalue(ax2, hl, y, row)

            ax2.set_xlabel("")
            ax2.set_title("")
            ax2.grid(axis="x", linestyle=":", linewidth=0.6, color="#bdbdbd", alpha=0.8)
            ax2.tick_params(axis="both", labelsize=14)
            for spine in ("top", "right"):
                ax2.spines[spine].set_visible(False)

            # Legend for overall figure
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
                fontsize=14,
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
    show_pvalues: bool = False,
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
    plot_paths = save_mean_difference_ci_plot(intervals, out_dir / name, show_pvalues=show_pvalues)
    return intervals, [csv_path, *plot_paths]

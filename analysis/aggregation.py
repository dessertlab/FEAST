"""Two-stage aggregation of per-family fusion metrics.

The experiment produces, for every fold, a table of metrics per ``(strategy, family)``.
The headline numbers are built in two explicit stages, as required:

* **Stage 1 — mean over folds**: for each ``(strategy, family)`` average every metric
  across the K folds. Family support (``positive_support``) is *summed* across folds, so
  it equals the family's total ground-truth occurrences in the dataset.
* **Stage 2 — support-weighted mean over families**: for each ``(strategy, metric)`` take
  the mean across families weighted by that family support. This is the headline value:
  it reflects how the ensemble does on the bulk of real positives. The unweighted macro
  mean and the median are kept alongside as context (a rare family weighs the same as a
  common one in the macro mean; the median is robust to the long tail).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

# Classification metrics aggregated for the report (per-family columns produced by
# analysis.fusion.predictions.binary_metrics).
REPORT_METRICS = (
    "precision", "recall", "specificity", "npv", "fpr", "fnr",
    "f1", "f2", "mcc", "accuracy", "balanced_accuracy", "roc_auc", "pr_auc",
)
SUPPORT_COLUMN = "positive_support"


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def mean_over_folds(
    per_family_per_fold: pd.DataFrame,
    metric_columns: Iterable[str] = REPORT_METRICS,
) -> pd.DataFrame:
    """Stage 1: average metrics across folds per (strategy, family); sum the support."""
    metrics = [m for m in metric_columns if m in per_family_per_fold.columns]
    agg = {m: "mean" for m in metrics}
    if SUPPORT_COLUMN in per_family_per_fold.columns:
        agg[SUPPORT_COLUMN] = "sum"  # total GT occurrences across the dataset
    out = (
        per_family_per_fold.groupby(["strategy", "family"], as_index=False, sort=True)
        .agg(agg)
    )
    return out


def support_weighted_over_families(
    per_family: pd.DataFrame,
    metric_columns: Iterable[str] = REPORT_METRICS,
    support_column: str = SUPPORT_COLUMN,
) -> pd.DataFrame:
    """Stage 2: per strategy, reduce families to weighted/macro/median per metric."""
    metrics = [m for m in metric_columns if m in per_family.columns]
    has_weight = support_column in per_family.columns

    rows: list[dict] = []
    for strategy, group in per_family.groupby("strategy", dropna=False, sort=True):
        record: dict = {"strategy": strategy, "n_families": int(len(group))}
        weights = pd.to_numeric(group[support_column], errors="coerce") if has_weight else None
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce")
            record[f"{metric}_macro"] = _safe_float(values.mean())
            record[f"{metric}_median"] = _safe_float(values.median())
            if weights is None:
                record[f"{metric}_weighted"] = None
                continue
            mask = values.notna() & weights.notna() & (weights > 0)
            total = float(weights[mask].sum())
            record[f"{metric}_weighted"] = (
                _safe_float((values[mask] * weights[mask]).sum() / total) if total > 0 else None
            )
        rows.append(record)
    return pd.DataFrame(rows)


def aggregate_fusion_metrics(
    per_family_per_fold: pd.DataFrame,
    metric_columns: Iterable[str] = REPORT_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run both stages; return (per_family_mean_over_folds, per_strategy_overall)."""
    per_family = mean_over_folds(per_family_per_fold, metric_columns)
    overall = support_weighted_over_families(per_family, metric_columns)
    return per_family, overall

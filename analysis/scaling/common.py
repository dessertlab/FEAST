"""Shared building blocks for the scaling / ablation studies.

Two jobs:

* read the *dataset-level covariates* that the cross-language comparison hinges on
  (number of tools, findings, families, coverage), from the ``config.json`` each fusion run
  already writes; and
* turn the raw ``fusion_metrics_per_family_per_fold.csv`` into the per-(family, fold)
  paired ``delta_f1`` against the traditional baseline — the unit of analysis for the
  meta-regression, with the family random-intercept structure preserved.

Nothing here re-runs fusion; it only reads the CSV/JSON outputs already on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from analysis.aggregation import SUPPORT_COLUMN
from analysis.mean_difference_ci import _resolve_baseline

# The three languages the pipeline supports, ordered small -> large tool pool.
LANGUAGES = ["java", "python", "c_cpp"]
LEVEL = "pillar_child"


def results_dir(language: str, tier: str = "full", *, level: str = LEVEL,
                results_root: str | Path = "data/results") -> Path:
    """Directory holding one fusion run's CSV/JSON outputs."""
    return Path(results_root) / language / level / tier


def load_config(language: str, tier: str = "full", *, level: str = LEVEL,
                results_root: str | Path = "data/results") -> dict:
    path = results_dir(language, tier, level=level, results_root=results_root) / "config.json"
    if not path.exists():
        raise FileNotFoundError(f"missing fusion config: {path} (run `python main.py fusion` first)")
    return json.loads(path.read_text(encoding="utf-8"))


def n_samples(language: str, tier: str = "full", *, level: str = LEVEL,
              results_root: str | Path = "data/results") -> int:
    """Number of code samples that entered the run (rows of ``folds.csv``)."""
    path = results_dir(language, tier, level=level, results_root=results_root) / "folds.csv"
    if not path.exists():
        return -1
    return int(len(pd.read_csv(path)))


def dataset_covariates(language: str, tier: str = "full", *, level: str = LEVEL,
                       results_root: str | Path = "data/results") -> dict:
    """Per-language predictors for the cross-language model.

    ``n_tools`` and the restriction counts come straight from the run's ``config.json``;
    ``n_samples`` from ``folds.csv``.  These are the language-level covariates whose effect
    on ``delta_f1`` we want to disentangle from the bare language label.
    """
    cfg = load_config(language, tier, level=level, results_root=results_root)
    r = cfg.get("restriction", {})
    return {
        "language": language,
        "tools": cfg.get("tools", []),
        "n_tools": len(cfg.get("tools", [])),
        "n_samples": n_samples(language, tier, level=level, results_root=results_root),
        "gt_kept": r.get("gt_occurrences_kept"),
        "gt_total": r.get("gt_occurrences_total"),
        "kept_frac": r.get("gt_occurrences_kept_frac"),
        "families_kept": r.get("families_kept"),
        "families_present": r.get("families_present"),
    }


def load_per_family_per_fold(language: str, tier: str = "full", *, level: str = LEVEL,
                             results_root: str | Path = "data/results") -> pd.DataFrame:
    path = (results_dir(language, tier, level=level, results_root=results_root)
            / "fusion_metrics_per_family_per_fold.csv")
    if not path.exists():
        raise FileNotFoundError(f"missing per-fold metrics: {path}")
    return pd.read_csv(path)


def fold_weighted_f1(per_family_per_fold: pd.DataFrame, strategy: str,
                     metric: str = "f1") -> pd.Series:
    """Support-weighted ``metric`` per fold for one strategy (indexed by fold).

    Same aggregation the headline 'overall (support-weighted)' table uses, exposed here so
    the ablation curves can report an *absolute* level, not only a difference.
    """
    sub = per_family_per_fold[per_family_per_fold["strategy"] == strategy]
    out: dict = {}
    for fold, group in sub.groupby("fold"):
        values = pd.to_numeric(group[metric], errors="coerce")
        weights = pd.to_numeric(group[SUPPORT_COLUMN], errors="coerce")
        mask = values.notna() & weights.notna() & (weights > 0)
        total = float(weights[mask].sum())
        out[fold] = float((values[mask] * weights[mask]).sum() / total) if total > 0 else float("nan")
    return pd.Series(out, name=metric)


def overall_f1_by_strategy(per_family_per_fold: pd.DataFrame, metric: str = "f1") -> pd.Series:
    """Mean-over-folds of the support-weighted ``metric``, one value per strategy.

    Reproduces the headline 'overall (support-weighted)' number without re-reading the
    aggregated CSV, so it works on any (e.g. ablation) run held only in memory or in a
    scratch directory.
    """
    out = {
        strategy: float(fold_weighted_f1(per_family_per_fold, strategy, metric).mean())
        for strategy in per_family_per_fold["strategy"].astype(str).unique()
    }
    return pd.Series(out, name=metric).sort_values(ascending=False)


def run_outcomes(per_family_per_fold: pd.DataFrame, *, metric: str = "f1",
                 baseline: str = "traditional") -> dict:
    """Collapse one fusion run to the handful of numbers the ablation curves plot.

    Avoids post-hoc cherry-picking by reporting the *mean* over fusion strategies alongside
    the max (the achievable ceiling) and a couple of named strategies, all as support-weighted
    f1, plus the best fusion strategy's lift over the traditional baseline.
    """
    f1 = overall_f1_by_strategy(per_family_per_fold, metric)
    present = set(per_family_per_fold["strategy"].astype(str))
    base_name = _resolve_baseline(present, baseline)

    fusion = f1.drop(labels=[s for s in f1.index if s.startswith(("tool:", "or_", "traditional"))],
                     errors="ignore")
    bks = f1[[s for s in f1.index if s.startswith("bks")]]
    best_fusion = fusion.index[0] if len(fusion) else None
    return {
        "n_strategies": int(len(present)),
        "baseline_strategy": base_name,
        "traditional_f1": float(f1.get(base_name, float("nan"))),
        "or_f1": float(next((f1[s] for s in f1.index if s.startswith("or_")), float("nan"))),
        "fusion_f1_mean": float(fusion.mean()) if len(fusion) else float("nan"),
        "fusion_f1_max": float(fusion.max()) if len(fusion) else float("nan"),
        "best_fusion_strategy": best_fusion,
        "bks_f1": float(bks.max()) if len(bks) else float("nan"),
        "best_fusion_minus_traditional": (
            float(fusion.max() - f1.get(base_name, float("nan"))) if len(fusion) else float("nan")
        ),
    }


def paired_delta_per_family_fold(
    per_family_per_fold: pd.DataFrame,
    *,
    metric: str = "f1",
    baseline: str = "traditional",
    strategies: list[str] | None = None,
) -> pd.DataFrame:
    """Per-(family, fold) paired difference ``metric(strategy) - metric(baseline)``.

    Returns one row per (strategy, fold, family) with the family ``support`` retained, so a
    downstream mixed model can put a random intercept on ``family``.  ``baseline`` is resolved
    the same way as in ``mean_difference_ci`` (e.g. ``"traditional"`` -> the single
    ``traditional_K_of_N`` strategy present).  Pass ``strategies`` to restrict to specific
    fusers; by default every non-baseline strategy is included.
    """
    present = set(per_family_per_fold["strategy"].astype(str))
    base_name = _resolve_baseline(present, baseline)

    base = (
        per_family_per_fold[per_family_per_fold["strategy"] == base_name]
        [["fold", "family", metric, SUPPORT_COLUMN]]
        .rename(columns={metric: "baseline_value"})
    )
    keep = [s for s in present if s != base_name]
    if strategies is not None:
        keep = [s for s in keep if s in set(strategies)]

    frames = []
    for strategy in keep:
        strat = (
            per_family_per_fold[per_family_per_fold["strategy"] == strategy]
            [["fold", "family", metric]]
            .rename(columns={metric: "strategy_value"})
        )
        merged = strat.merge(base, on=["fold", "family"], how="inner")
        merged["strategy"] = strategy
        merged["baseline_strategy"] = base_name
        merged["metric"] = metric
        merged["delta"] = (
            pd.to_numeric(merged["strategy_value"], errors="coerce")
            - pd.to_numeric(merged["baseline_value"], errors="coerce")
        )
        frames.append(merged)

    if not frames:
        return pd.DataFrame(
            columns=["fold", "family", SUPPORT_COLUMN, "strategy", "baseline_strategy",
                     "metric", "strategy_value", "baseline_value", "delta"]
        )
    out = pd.concat(frames, ignore_index=True)
    return out[["fold", "family", SUPPORT_COLUMN, "strategy", "baseline_strategy",
                "metric", "strategy_value", "baseline_value", "delta"]]

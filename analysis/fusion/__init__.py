"""Fusion strategies operating on canonical CWE families with exact matching.

``run_fusion`` runs every strategy on one fold and returns a single long prediction frame
(schema ``common.PREDICTION_COLUMNS``). Labels, the validation fire index and the
reliability lookup are computed once and shared across strategies.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from analysis.fusion.baselines import or_predictions, single_tool_predictions
from analysis.fusion.bayes import naive_bayes_predictions
from analysis.fusion.bks import bks_predictions
from analysis.fusion.common import (
    metric_pairs_for_calibration,
    PREDICTION_COLUMNS,
    build_fire_index,
    canonical_strategy_order,
    metric_lookup,
    order_by_strategy,
    precompute_labels,
)
from analysis.fusion.dst import dst_vote_predictions
from analysis.fusion.logistic import logistic_regression_predictions
from analysis.fusion.predictions import (
    detection_from_predictions,
    detection_metrics,
    evaluate_predictions,
    expand_tau_variants,
)
from analysis.fusion.traditional import traditional_vote_predictions
from analysis.fusion.weighted import DEFAULT_WEIGHTED_STRATEGIES, WeightedVotingStrategy, weighted_vote_predictions

__all__ = [
    "run_fusion",
    "evaluate_predictions",
    "detection_from_predictions",
    "detection_metrics",
    "order_by_strategy",
    "canonical_strategy_order",
    "DEFAULT_WEIGHTED_STRATEGIES",
    "PREDICTION_COLUMNS",
]


def run_fusion(
    calibration_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    reliability: pd.DataFrame,
    tools: Sequence[str],
    families: Sequence[str],
    threshold: int = 2,
    seed: int = 0,
    include_logistic_regression: bool = True,
    calibration_metrics: Sequence[str] | None = None,
    taus: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Run all fusion strategies on one fold's validation split.

    ``reliability`` is the per-(tool, family) calibration table; ``calibration_df`` is only
    needed by the strategies that learn (logistic regression).
    """
    metric_pairs = metric_pairs_for_calibration(calibration_metrics)
    weighted_strategies = tuple(WeightedVotingStrategy(fire, silence) for fire, silence in metric_pairs)

    lookup = metric_lookup(reliability)
    labels = precompute_labels(validation_df, families)
    fire_index = build_fire_index(validation_df, tools)

    baseline_frames = [
        # References (shown first): each single tool, then their OR (1-of-N).
        *(single_tool_predictions(validation_df, t, families, labels, fire_index) for t in tools),
        or_predictions(validation_df, tools, families, labels, fire_index),
        # Baseline: traditional K-of-N over all tools.
        traditional_vote_predictions(validation_df, tools, families, threshold, labels, fire_index),
    ]
    scored_fuser_frames = [
        # Reliability-weighted voting (one frame per metric pair).
        *(weighted_vote_predictions(validation_df, lookup, strategy, tools, families, labels, fire_index)
          for strategy in weighted_strategies),
        # Evidence-theoretic: each rule expanded over the same fire/silence metric pairs.
        *(
            dst_vote_predictions(
                validation_df, lookup, rule, tools, families, labels, fire_index,
                fire_metric=fire_metric, silence_metric=silence_metric,
            )
            for rule in ("dempster", "pcr6", "yager")
            for fire_metric, silence_metric in metric_pairs
        ),
        # Probabilistic and pattern-learning fusers.
        naive_bayes_predictions(validation_df, lookup, tools, families, labels, fire_index),
        bks_predictions(calibration_df, validation_df, tools, families, labels, fire_index),
    ]
    if include_logistic_regression:
        scored_fuser_frames.append(
            logistic_regression_predictions(calibration_df, validation_df, tools, families, labels, fire_index, seed=seed))
        scored_fuser_frames.append(
            logistic_regression_predictions(calibration_df, validation_df, tools, families, labels, fire_index, seed=seed, interactions=True))
    scored_fusers = expand_tau_variants(pd.concat(scored_fuser_frames, ignore_index=True), taus=taus)
    return pd.concat([*baseline_frames, scored_fusers], ignore_index=True)

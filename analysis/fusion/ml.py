"""ML-based fusion strategies: Decision Tree, Random Forest, Gradient Boosting.

All three learn a per-family binary classifier (one-vs-rest) on the calibration split
using the tool fire indicators as features.  The same four-feature vector used by
logistic_regression is used here:

    x[i] = [bandit_fired, codeql_fired, pylint_fired, semgrep_fired]  (0/1 floats)

Class imbalance is handled without sampling (SMOTE is inapplicable on binary features):
  - DecisionTreeClassifier / RandomForestClassifier: class_weight='balanced'
  - GradientBoostingClassifier: sample_weight proportional to inverse class frequency
    (sklearn GradientBoostingClassifier does not accept class_weight natively)

Hyperparameter rationale (all choices adapted from D'Abruzzo Pereira et al., 2024 to
the 4-binary-feature, 30–900 samples/family regime of FEAST):

  Decision Tree — tier: medium (≥30 GT occurrences)
    max_depth=2        paper does not set this (high-dim data); depth 2 covers all
                       pairwise interactions without memorising the calibration fold
    min_samples_leaf=3 paper tests 0.001 ≈ 0 on 8k rows; on calibration sets of
                       24–80 rows this produces single-sample leaves
    min_samples_split=5 same motivation: prevents noisy micro-splits
    class_weight='balanced'  replaces SMOTE/oversampling (inapplicable on binary space)

  Random Forest — tier: full (≥100 GT occurrences)
    n_estimators=100   paper tests up to 200; convergence is fast on 16 distinct states
    max_features=2     paper uses 0.55 ≈ 2 features out of 4 (same as sqrt rule)
    max_depth=3        one level deeper than DT to allow more ensemble diversity
    min_samples_leaf=3
    min_samples_split=5
    class_weight='balanced'

  Gradient Boosting — tier: full (≥100 GT occurrences)
    n_estimators=50    paper tests up to 200; 50 rounds suffice on a 16-state space
    learning_rate=0.1  paper tests [0.1, 1.0]; conservative value is mandatory here
    max_depth=2        base learners in GB should be weak (depth 2 = one interaction)
    max_features=2     stochastic GB: adds regularisation through feature subsampling
    min_samples_leaf=3
    min_samples_split=5
    subsample=0.8      not in paper; stochastic row subsampling further reduces overfit
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from analysis.fusion.common import (
    build_fire_index,
    evidence_row,
    precompute_labels,
    sample_ids_of,
)


def _build_feature_matrix(
    fire: dict[tuple[int, str], set[str]],
    n_rows: int,
    tools: Sequence[str],
    family: str,
) -> np.ndarray:
    return np.array(
        [[float(family in fire[(i, t)]) for t in tools] for i in range(n_rows)],
        dtype=float,
    )


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency sample weights, equivalent to class_weight='balanced'."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    pos_w = n_neg / max(n_pos, 1)
    return np.where(y == 1, pos_w, 1.0)


def _fallback_score(y_cal: np.ndarray, n_val: int) -> np.ndarray:
    """Constant score equal to the calibration positive rate (used when a family
    is single-class in calibration — the classifier cannot be fitted)."""
    return np.full(n_val, float(y_cal.mean()) if len(y_cal) else 0.0)


# ── Decision Tree ─────────────────────────────────────────────────────────────

def decision_tree_predictions(
    calibration_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    tools: Sequence[str],
    families: Sequence[str],
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
    tau: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    from sklearn.tree import DecisionTreeClassifier

    cal_fire, _ = build_fire_index(calibration_df, tools)
    val_fire, _ = fire_index
    cal_labels = precompute_labels(calibration_df, families)
    sample_ids = sample_ids_of(validation_df)
    n_cal, n_val = len(calibration_df), len(validation_df)

    rows: list[dict] = []
    for family in families:
        x_cal = _build_feature_matrix(cal_fire, n_cal, tools, family)
        y_cal = np.array([int(cal_labels[(i, family)]) for i in range(n_cal)])
        x_val = _build_feature_matrix(val_fire, n_val, tools, family)
        fired_counts = x_val.sum(axis=1).astype(int)

        if len(np.unique(y_cal)) < 2:
            score = _fallback_score(y_cal, n_val)
        else:
            clf = DecisionTreeClassifier(
                criterion="gini",
                max_depth=2,
                min_samples_leaf=3,
                min_samples_split=5,
                class_weight="balanced",
                random_state=seed,
            )
            clf.fit(x_cal, y_cal)
            score = clf.predict_proba(x_val)[:, 1]

        for ri in range(n_val):
            rows.append(evidence_row(
                sample_ids[ri], ri, family, "decision_tree",
                prediction=bool(score[ri] >= tau), score=float(score[ri]),
                vuln=float(score[ri]), safe=float(1.0 - score[ri]),
                k_supported=len(tools), k_fired=int(fired_counts[ri]), k_abstained=0,
                label=labels[(ri, family)],
            ))
    return pd.DataFrame(rows)


# ── Random Forest ─────────────────────────────────────────────────────────────

def random_forest_predictions(
    calibration_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    tools: Sequence[str],
    families: Sequence[str],
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
    tau: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestClassifier

    cal_fire, _ = build_fire_index(calibration_df, tools)
    val_fire, _ = fire_index
    cal_labels = precompute_labels(calibration_df, families)
    sample_ids = sample_ids_of(validation_df)
    n_cal, n_val = len(calibration_df), len(validation_df)
    max_features = min(2, len(tools))

    rows: list[dict] = []
    for family in families:
        x_cal = _build_feature_matrix(cal_fire, n_cal, tools, family)
        y_cal = np.array([int(cal_labels[(i, family)]) for i in range(n_cal)])
        x_val = _build_feature_matrix(val_fire, n_val, tools, family)
        fired_counts = x_val.sum(axis=1).astype(int)

        if len(np.unique(y_cal)) < 2:
            score = _fallback_score(y_cal, n_val)
        else:
            clf = RandomForestClassifier(
                n_estimators=100,
                max_features=max_features,
                max_depth=3,
                min_samples_leaf=3,
                min_samples_split=5,
                bootstrap=True,
                class_weight="balanced",
                random_state=seed,
            )
            clf.fit(x_cal, y_cal)
            score = clf.predict_proba(x_val)[:, 1]

        for ri in range(n_val):
            rows.append(evidence_row(
                sample_ids[ri], ri, family, "random_forest",
                prediction=bool(score[ri] >= tau), score=float(score[ri]),
                vuln=float(score[ri]), safe=float(1.0 - score[ri]),
                k_supported=len(tools), k_fired=int(fired_counts[ri]), k_abstained=0,
                label=labels[(ri, family)],
            ))
    return pd.DataFrame(rows)


# ── Gradient Boosting ─────────────────────────────────────────────────────────

def gradient_boosting_predictions(
    calibration_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    tools: Sequence[str],
    families: Sequence[str],
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
    tau: float = 0.5,
    seed: int = 0,
) -> pd.DataFrame:
    from sklearn.ensemble import GradientBoostingClassifier

    cal_fire, _ = build_fire_index(calibration_df, tools)
    val_fire, _ = fire_index
    cal_labels = precompute_labels(calibration_df, families)
    sample_ids = sample_ids_of(validation_df)
    n_cal, n_val = len(calibration_df), len(validation_df)
    max_features = min(2, len(tools))

    rows: list[dict] = []
    for family in families:
        x_cal = _build_feature_matrix(cal_fire, n_cal, tools, family)
        y_cal = np.array([int(cal_labels[(i, family)]) for i in range(n_cal)])
        x_val = _build_feature_matrix(val_fire, n_val, tools, family)
        fired_counts = x_val.sum(axis=1).astype(int)

        if len(np.unique(y_cal)) < 2:
            score = _fallback_score(y_cal, n_val)
        else:
            clf = GradientBoostingClassifier(
                n_estimators=50,
                learning_rate=0.1,
                max_depth=2,
                max_features=max_features,
                min_samples_leaf=3,
                min_samples_split=5,
                subsample=0.8,
                random_state=seed,
            )
            clf.fit(x_cal, y_cal, sample_weight=_balanced_sample_weight(y_cal))
            score = clf.predict_proba(x_val)[:, 1]

        for ri in range(n_val):
            rows.append(evidence_row(
                sample_ids[ri], ri, family, "gradient_boosting",
                prediction=bool(score[ri] >= tau), score=float(score[ri]),
                vuln=float(score[ri]), safe=float(1.0 - score[ri]),
                k_supported=len(tools), k_fired=int(fired_counts[ri]), k_abstained=0,
                label=labels[(ri, family)],
            ))
    return pd.DataFrame(rows)

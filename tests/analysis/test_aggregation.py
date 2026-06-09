import pandas as pd
from pytest import approx

from analysis.aggregation import aggregate_fusion_metrics, mean_over_folds, support_weighted_over_families


def _per_family_per_fold():
    # One strategy, two families, two folds. f1 metric chosen so the maths is checkable.
    return pd.DataFrame([
        {"fold": 0, "strategy": "S", "family": "CWE-1", "f1": 0.8, "positive_support": 8},
        {"fold": 1, "strategy": "S", "family": "CWE-1", "f1": 0.6, "positive_support": 8},
        {"fold": 0, "strategy": "S", "family": "CWE-2", "f1": 0.2, "positive_support": 2},
        {"fold": 1, "strategy": "S", "family": "CWE-2", "f1": 0.4, "positive_support": 2},
    ])


def test_stage1_means_metrics_and_sums_support():
    per_family = mean_over_folds(_per_family_per_fold(), metric_columns=("f1",))
    by_family = per_family.set_index("family")
    assert by_family.at["CWE-1", "f1"] == approx(0.7)   # mean(0.8, 0.6)
    assert by_family.at["CWE-2", "f1"] == approx(0.3)   # mean(0.2, 0.4)
    assert by_family.at["CWE-1", "positive_support"] == 16  # summed across folds
    assert by_family.at["CWE-2", "positive_support"] == 4


def test_stage2_support_weighted_macro_median():
    per_family = mean_over_folds(_per_family_per_fold(), metric_columns=("f1",))
    overall = support_weighted_over_families(per_family, metric_columns=("f1",)).iloc[0]
    # weighted by summed support 16 vs 4: (0.7*16 + 0.3*4) / 20 = 0.62
    assert abs(overall["f1_weighted"] - 0.62) < 1e-9
    assert abs(overall["f1_macro"] - 0.5) < 1e-9        # mean(0.7, 0.3)
    assert abs(overall["f1_median"] - 0.5) < 1e-9


def test_aggregate_fusion_metrics_returns_both_stages():
    per_family, overall = aggregate_fusion_metrics(_per_family_per_fold(), metric_columns=("f1",))
    assert set(per_family["family"]) == {"CWE-1", "CWE-2"}
    assert "f1_weighted" in overall.columns and len(overall) == 1

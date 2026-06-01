
import pandas as pd

from analysis.fusion import (
    WeightedVotingStrategy,
    all_fusion_predictions,
    evaluate_predictions,
    traditional_vote_predictions,
    weighted_vote_predictions,
)


def _metrics():
    return pd.DataFrame([
        {
            "tool": "bandit", "cwe": "CWE-79", "tp": 1, "fp": 0, "tn": 2, "fn": 1,
            "ppv": 1.0, "npv": 2 / 3, "fpr": 0.0, "fnr": 0.5, "supported": True,
        },
        {
            "tool": "semgrep", "cwe": "CWE-79", "tp": 0, "fp": 0, "tn": 2, "fn": 2,
            "ppv": None, "npv": 0.5, "fpr": 0.0, "fnr": 1.0, "supported": False,
        },
    ])


def test_weighted_voting_unsupported_tools_abstain_even_when_they_fire():
    df = pd.DataFrame([
        {"sample_id": "a", "cwes": ["CWE-79"], "bandit": [], "semgrep": ["CWE-79"]},
    ])

    predictions = weighted_vote_predictions(
        df,
        _metrics(),
        WeightedVotingStrategy("ppv", "npv"),
        tools=["bandit", "semgrep"],
        cwes=["CWE-79"],
    )

    row = predictions.iloc[0]
    assert row["tools_supported"] == 1
    assert row["tools_abstained"] == 1
    assert row["tools_fired"] == 0
    assert row["safe_score"] == 2 / 3
    assert row["vuln_score"] == 0
    assert bool(row["prediction"]) is False


def test_traditional_voting_can_run_fixed_or_supported_only_baseline():
    df = pd.DataFrame([
        {"sample_id": "a", "cwes": ["CWE-79"], "bandit": ["CWE-79"], "semgrep": ["CWE-79"]},
    ])

    fixed = traditional_vote_predictions(df, tools=["bandit", "semgrep"], cwes=["CWE-79"], threshold=2)
    supported = traditional_vote_predictions(
        df,
        tools=["bandit", "semgrep"],
        cwes=["CWE-79"],
        threshold=1,
        metrics=_metrics(),
        supported_only=True,
    )

    assert bool(fixed.iloc[0]["prediction"]) is True
    assert fixed.iloc[0]["tools_supported"] == 2
    assert bool(supported.iloc[0]["prediction"]) is True
    assert supported.iloc[0]["tools_supported"] == 1
    assert supported.iloc[0]["tools_abstained"] == 1


def test_evaluate_predictions_reports_f_scores_and_auc():
    predictions = pd.DataFrame([
        {"strategy": "s", "cwe": "CWE-79", "label": True, "prediction": True, "score": 0.9},
        {"strategy": "s", "cwe": "CWE-79", "label": False, "prediction": False, "score": 0.1},
        {"strategy": "s", "cwe": "CWE-79", "label": True, "prediction": True, "score": 0.8},
        {"strategy": "s", "cwe": "CWE-79", "label": False, "prediction": False, "score": 0.2},
    ])

    metrics = evaluate_predictions(predictions)

    row = metrics.iloc[0]
    assert row["f1"] == 1.0
    assert row["f2"] == 1.0
    assert row["roc_auc"] == 1.0
    assert row["pr_auc"] == 1.0


def test_all_fusion_predictions_includes_traditional_and_weighted_strategies():
    df = pd.DataFrame([
        {"sample_id": "a", "cwes": ["CWE-79"], "bandit": ["CWE-79"], "semgrep": []},
    ])

    predictions = all_fusion_predictions(
        df,
        _metrics(),
        tools=["bandit", "semgrep"],
        cwes=["CWE-79"],
        strategies=(WeightedVotingStrategy("ppv", "npv"),),
    )

    assert set(predictions["strategy"]) == {
        "traditional_2_of_2",
        "traditional_2_of_supported",
        "weighted_fire_ppv_silence_npv",
    }

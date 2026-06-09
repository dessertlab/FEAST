import numpy as np
import pandas as pd
from pytest import approx

from analysis.calibration import compute_reliability
from analysis.fusion import run_fusion
from analysis.fusion.common import build_fire_index, metric_lookup, precompute_labels
from analysis.fusion.bks import bks_predictions
from analysis.fusion.dst import _combine_dempster, _combine_yager
from analysis.fusion.noisy_or import noisy_or_predictions
from analysis.fusion.predictions import DEFAULT_TAUS, tau_sweep_detection


def _df():
    return pd.DataFrame([
        {"sample_id": "a", "label": 1, "cwes": ["CWE-1"], "toolA": ["CWE-1"], "toolB": ["CWE-1"]},
        {"sample_id": "b", "label": 1, "cwes": ["CWE-1"], "toolA": ["CWE-1"], "toolB": []},
        {"sample_id": "c", "label": 0, "cwes": [],        "toolA": [],        "toolB": []},
        {"sample_id": "d", "label": 1, "cwes": ["CWE-2"], "toolA": ["CWE-2"], "toolB": ["CWE-2"]},
        {"sample_id": "e", "label": 0, "cwes": [],        "toolA": ["CWE-1"], "toolB": []},
    ])


def _ctx(df, tools=("toolA", "toolB"), families=("CWE-1", "CWE-2")):
    tools, families = list(tools), list(families)
    rel = compute_reliability(df, tools, families)
    return tools, families, metric_lookup(rel), precompute_labels(df, families), build_fire_index(df, tools), rel


def test_run_fusion_includes_all_new_strategies():
    df = _df()
    tools, families, _, _, _, rel = _ctx(df)
    preds = run_fusion(df, df, rel, tools, families, threshold=2)
    strategies = set(preds["strategy"])
    for expected in {"bks", "noisy_or", "dst_yager", "logistic_interactions", "naive_bayes"}:
        assert expected in strategies


def test_noisy_or_is_monotone_in_fires():
    # more firing tools -> score never decreases (multiplicative OR)
    df = _df()
    tools, families, lookup, labels, fire_index, _ = _ctx(df)
    preds = noisy_or_predictions(df, lookup, tools, families, labels, fire_index)
    f1 = preds[preds["family"] == "CWE-1"].set_index("row_index")["score"]
    # row 0 (both tools fire CWE-1) >= row 1 (only toolA fires CWE-1) >= row 2 (none)
    assert f1[0] >= f1[1] >= f1[2]


def test_bks_scores_are_probabilities_and_cover_all_rows():
    df = _df()
    tools, families, _, labels, fire_index, _ = _ctx(df)
    preds = bks_predictions(df, df, tools, families, labels, fire_index)
    assert ((preds["score"] >= 0) & (preds["score"] <= 1)).all()
    assert len(preds) == len(df) * len(families)


def test_yager_unnormalised_mass_le_dempster():
    # On conflict, Yager leaves mass on ignorance: mV+mS <= the normalised Dempster masses.
    v = np.array([[0.6, 0.0]]); s = np.array([[0.0, 0.7]]); t = 1.0 - v - s
    yV, yS = _combine_yager(v, s, t)
    dV, dS = _combine_dempster(v, s, t)
    assert yV + yS <= dV + dS + 1e-9


def test_canonical_strategy_order():
    from analysis.fusion.common import canonical_strategy_order
    present = ["bks", "tool:codeql", "logistic_regression", "or_1_of_2",
               "tool:bandit", "traditional_2_of_2", "naive_bayes", "always_vulnerable"]
    order = canonical_strategy_order(["bandit", "codeql"], present)
    # single tools first (in tool order), then OR, then traditional, then fusers, unknown last
    assert order[:2] == ["tool:bandit", "tool:codeql"]
    assert order[2] == "or_1_of_2"
    assert order[3] == "traditional_2_of_2"
    assert order.index("naive_bayes") < order.index("logistic_regression")
    assert order[-1] == "always_vulnerable"


def test_run_fusion_includes_single_tools_and_or():
    df = _df()
    tools, families, _, _, _, rel = _ctx(df)
    strategies = set(run_fusion(df, df, rel, tools, families)["strategy"])
    assert {"tool:toolA", "tool:toolB", "or_1_of_2"} <= strategies


def test_tau_sweep_grid_and_baseline():
    df = _df()
    tools, families, _, _, _, rel = _ctx(df)
    preds = run_fusion(df, df, rel, tools, families, threshold=2)
    sweep = tau_sweep_detection(preds, df)
    # one row per (scored strategy, tau) + the always_vulnerable baseline
    assert set(sweep["tau"]).issuperset(set(DEFAULT_TAUS))
    assert "always_vulnerable" in set(sweep["strategy"])
    always = sweep[sweep["strategy"] == "always_vulnerable"].iloc[0]
    assert always["recall"] == approx(1.0)   # predicts every row vulnerable

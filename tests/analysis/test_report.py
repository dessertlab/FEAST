
import pandas as pd

from analysis.report import aggregate_by_strategy, save_fusion_report


def test_aggregate_by_strategy_means_metrics():
    metrics = pd.DataFrame([
        {"strategy": "a", "f1": 0.5, "f2": 0.7},
        {"strategy": "a", "f1": 1.0, "f2": 0.9},
        {"strategy": "b", "f1": 0.25, "f2": 0.5},
    ])

    out = aggregate_by_strategy(metrics, metric_columns=("f1", "f2"))

    assert out[out["strategy"] == "a"]["f1"].item() == 0.75
    assert out[out["strategy"] == "b"]["f2"].item() == 0.5


def test_save_fusion_report_writes_csv_svg_and_html(tmp_path):
    metrics = pd.DataFrame([
        {"strategy": "traditional_2_of_5", "f1": 0.5, "f2": 0.6},
        {"strategy": "weighted_fire_ppv_silence_npv", "f1": 0.75, "f2": 0.8},
    ])

    paths = save_fusion_report(metrics, tmp_path, metric_columns=("f1", "f2"))

    assert paths["aggregate_csv"].exists()
    assert paths["f1_svg"].exists()
    assert paths["f2_svg"].exists()
    assert paths["html"].exists()

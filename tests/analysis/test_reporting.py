import pandas as pd

from analysis.aggregation import REPORT_METRICS
from analysis.reporting import best_variants_for_metric, save_best_variant_plots, save_strategy_plots


def test_strategy_plots_cover_every_report_metric_in_svg_and_png(tmp_path):
    row = {"strategy": "demo", **{f"{metric}_weighted": 0.5 for metric in REPORT_METRICS}}
    written = save_strategy_plots(pd.DataFrame([row]), tmp_path)

    written_names = {path.name for path in written}
    for metric in REPORT_METRICS:
        assert f"{metric}.svg" in written_names
        assert f"{metric}.png" in written_names
        assert (tmp_path / f"{metric}.svg").is_file()
        assert (tmp_path / f"{metric}.png").is_file()


def test_best_variants_are_selected_independently_per_validation_metric():
    overall = pd.DataFrame([
        {
            "strategy": "weighted_fire_ppv_silence_npv_tau_0_5",
            "f1_weighted": 0.80,
            "fpr_weighted": 0.30,
        },
        {
            "strategy": "weighted_fire_fpr_silence_fnr_tau_0_7",
            "f1_weighted": 0.70,
            "fpr_weighted": 0.10,
        },
        {
            "strategy": "dst_yager_fire_ppv_silence_npv_tau_0_5",
            "f1_weighted": 0.40,
            "fpr_weighted": 0.20,
        },
        {
            "strategy": "dst_yager_fire_fpr_silence_fnr_tau_0_3",
            "f1_weighted": 0.90,
            "fpr_weighted": 0.40,
        },
    ])

    best_f1 = best_variants_for_metric(overall, "f1")
    best_fpr = best_variants_for_metric(overall, "fpr")

    by_group_f1 = dict(zip(best_f1["strategy_group"], best_f1["strategy"], strict=False))
    by_group_fpr = dict(zip(best_fpr["strategy_group"], best_fpr["strategy"], strict=False))

    assert by_group_f1["weighted"] == "weighted_fire_ppv_silence_npv_tau_0_5"
    assert by_group_fpr["weighted"] == "weighted_fire_fpr_silence_fnr_tau_0_7"
    assert by_group_f1["dst_yager"] == "dst_yager_fire_fpr_silence_fnr_tau_0_3"
    assert by_group_fpr["dst_yager"] == "dst_yager_fire_ppv_silence_npv_tau_0_5"


def test_best_variant_plots_write_svg_and_png(tmp_path):
    overall = pd.DataFrame([
        {"strategy": "weighted_fire_ppv_silence_npv_tau_0_5", "f1_weighted": 0.8},
        {"strategy": "weighted_fire_fpr_silence_fnr_tau_0_7", "f1_weighted": 0.7},
        {"strategy": "naive_bayes_tau_0_4", "f1_weighted": 0.6},
    ])

    written = save_best_variant_plots(overall, tmp_path, metrics=("f1",))
    written_names = {path.name for path in written}

    assert written_names == {"f1.svg", "f1.png"}
    assert (tmp_path / "f1.svg").is_file()
    assert (tmp_path / "f1.png").is_file()
    assert "weighted (tau=0.5, cal=ppv/npv)" in (tmp_path / "f1.svg").read_text(encoding="utf-8")

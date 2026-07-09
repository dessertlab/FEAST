import pandas as pd

from analysis.mean_difference_ci import (
    STATUS_DEGRADATION,
    STATUS_IMPROVEMENT,
    STATUS_INCONCLUSIVE,
    mean_difference_ci,
    save_mean_difference_ci_report,
)


def _rows(strategy, values):
    return [
        {"strategy": strategy, "family": family, "f1": value}
        for family, value in zip(["CWE-1", "CWE-2", "CWE-3"], values, strict=False)
    ]


def _per_family():
    return pd.DataFrame([
        *_rows("or_1_of_4", [0.50, 0.50, 0.50]),
        *_rows("traditional_2_of_4", [0.50, 0.50, 0.50]),
        *_rows("weighted_fire_ppv_silence_npv_tau_0_5", [0.70, 0.80, 0.90]),
        *_rows("naive_bayes_tau_0_5", [0.60, 0.40, 0.50]),
        *_rows("bks_tau_0_5", [0.30, 0.20, 0.10]),
    ])


def test_mean_difference_ci_classifies_paired_intervals_against_traditional_baseline():
    intervals = mean_difference_ci(_per_family(), metric="f1", tools=["a", "b", "c", "d"])

    by_pair = {
        (row["baseline"], row["strategy"]): row["status"]
        for row in intervals.to_dict("records")
    }

    assert set(intervals["baseline"]) == {"traditional"}
    assert by_pair[("traditional", "weighted_fire_ppv_silence_npv_tau_0_5")] == STATUS_IMPROVEMENT
    assert by_pair[("traditional", "naive_bayes_tau_0_5")] == STATUS_INCONCLUSIVE
    assert by_pair[("traditional", "bks_tau_0_5")] == STATUS_DEGRADATION


def test_mean_difference_ci_report_writes_csv_svg_and_png(tmp_path):
    intervals, paths = save_mean_difference_ci_report(_per_family(), tmp_path, metric="f1")
    names = {path.name for path in paths}

    assert not intervals.empty
    assert names == {"mean_difference_ci_f1.csv", "mean_difference_ci_f1.svg", "mean_difference_ci_f1.png"}
    assert (tmp_path / "mean_difference_ci_f1.csv").is_file()
    assert (tmp_path / "mean_difference_ci_f1.svg").is_file()
    assert (tmp_path / "mean_difference_ci_f1.png").is_file()
    svg = (tmp_path / "mean_difference_ci_f1.svg").read_text(encoding="utf-8")
    assert "Weighted Voting" in svg
    assert "or_1_of_4" not in svg


from types import SimpleNamespace

import pandas as pd

from analysis.dataset import FeastDataset
from analysis.metrics import CWERelationshipMatcher, compute_tool_cwe_metrics, confusion_for_tool_cwe, save_metrics_outputs


def _matcher():
    # 79 and 89 are descendants of 20; 77 is a sibling of 79 under 20.
    def get_descendants(element_id):
        if element_id == "1347":
            return {"type": "Category", "by_view": {"1350": {1: ["79"]}}}
        if element_id == "1350":
            return {"type": "View", "by_view": {"1350": {1: ["1347"], 2: ["79"]}}}
        return {"type": "Weakness", "by_view": {}}

    return CWERelationshipMatcher(SimpleNamespace(
        child_of={"79": ["20"], "89": ["79"], "77": ["20"]},
        get_descendants=get_descendants,
    ))


def test_vertical_match_allows_ancestor_descendant_but_not_cousins():
    matcher = _matcher()

    assert matcher.is_vertical_match("CWE-20", "CWE-89")
    assert matcher.is_vertical_match("CWE-89", "CWE-20")
    assert matcher.is_vertical_match("CWE-79", "CWE-79")
    assert not matcher.is_vertical_match("CWE-79", "CWE-77")
    assert matcher.is_vertical_match("CWE-1347", "CWE-89")
    assert matcher.is_vertical_match("CWE-1350", "CWE-89")
    assert not matcher.is_vertical_match("CWE-1347", "CWE-77")


def test_confusion_for_tool_cwe_uses_exact_tool_target_and_tolerant_gt():
    df = pd.DataFrame([
        {"language": "Python", "cwes": ["CWE-20"], "bandit": ["CWE-89"]},  # TP: tool descendant of GT
        {"language": "Python", "cwes": ["CWE-77"], "bandit": ["CWE-89"]},  # FP: cousin, not vertical
        {"language": "Python", "cwes": ["CWE-89"], "bandit": []},          # FN
        {"language": "Python", "cwes": ["CWE-77"], "bandit": []},          # TN
    ])

    confusion = confusion_for_tool_cwe(df, "bandit", "CWE-89", _matcher())

    assert confusion == {"tool": "bandit", "cwe": "CWE-89", "tp": 1, "fp": 1, "tn": 1, "fn": 1, "total": 4}


def test_confusion_for_category_tool_cwe_expands_members_and_descendants():
    df = pd.DataFrame([
        {"language": "Python", "cwes": ["CWE-89"], "devaic": ["CWE-1347"]},  # TP: 89 under category member 79
        {"language": "Python", "cwes": ["CWE-77"], "devaic": ["CWE-1347"]},  # FP: sibling/cousin not in category
        {"language": "Python", "cwes": ["CWE-89"], "devaic": []},             # FN
        {"language": "Python", "cwes": ["CWE-77"], "devaic": []},             # TN
    ])

    confusion = confusion_for_tool_cwe(df, "devaic", "CWE-1347", _matcher())

    assert confusion == {"tool": "devaic", "cwe": "CWE-1347", "tp": 1, "fp": 1, "tn": 1, "fn": 1, "total": 4}


def test_compute_tool_cwe_metrics_uses_label_and_tool_output_universe(monkeypatch):
    dataset = FeastDataset(pd.DataFrame([
        {"language": "Python", "cwes": ["CWE-20"], "bandit": ["CWE-89"]},
        {"language": "Python", "cwes": ["CWE-77"], "bandit": []},
    ]))
    monkeypatch.setattr("analysis.metrics.CWERelationshipMatcher.from_xml", lambda _path: _matcher())

    payload = compute_tool_cwe_metrics(dataset, cwe_xml_path="unused.xml")

    by_cwe = {row["cwe"]: row for row in payload["metrics"]}
    assert list(by_cwe) == ["CWE-20", "CWE-77", "CWE-89"]

    supported = by_cwe["CWE-89"]
    assert supported["tp"] == 1
    assert supported["fp"] == 0
    assert supported["tn"] == 1
    assert supported["fn"] == 0
    assert supported["ppv"] == 1.0
    assert supported["fnr"] == 0.0
    assert supported["supported"] is True

    unsupported = by_cwe["CWE-77"]
    assert unsupported["tp"] == 0
    assert unsupported["fp"] == 0
    assert unsupported["supported"] is False



def test_save_metrics_outputs_writes_json_and_csv(tmp_path, monkeypatch):
    dataset = FeastDataset(pd.DataFrame([
        {"language": "Python", "cwes": ["CWE-20"], "bandit": ["CWE-89"]},
    ]))
    monkeypatch.setattr("analysis.metrics.CWERelationshipMatcher.from_xml", lambda _path: _matcher())
    payload = compute_tool_cwe_metrics(dataset, cwe_xml_path="unused.xml")

    json_path, csv_path = save_metrics_outputs(
        payload,
        json_path=tmp_path / "metrics.json",
        csv_path=tmp_path / "metrics.csv",
    )

    assert json_path.exists()
    assert csv_path.exists()
    csv_df = pd.read_csv(csv_path)
    assert list(csv_df.columns) == ["language", "tool", "cwe", "tp", "fp", "tn", "fn", "total", "ppv", "npv", "fpr", "fnr", "supported"]
    assert set(csv_df["cwe"]) == {"CWE-20", "CWE-89"}
    assert csv_df.loc[csv_df["cwe"] == "CWE-89", "supported"].item()

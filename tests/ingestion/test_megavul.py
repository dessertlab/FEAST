import json
import pytest
from pathlib import Path
from ingestion.megavul import extract_megavul
from ingestion.schema import FunctionSample


FIXTURE = [
    # Single-function commit, CWE present, high CVSS → valid positive
    {"func_before": "void vuln() {}", "CWE ID": "CWE-119",
     "commit_id": "aaa", "CVSS Score": 9.8, "target": 1},
    # Multi-function commit → filtered out
    {"func_before": "void vuln2() {}", "CWE ID": "CWE-787",
     "commit_id": "bbb", "CVSS Score": 7.5, "target": 1},
    {"func_before": "void vuln3() {}", "CWE ID": "CWE-119",
     "commit_id": "bbb", "CVSS Score": 7.5, "target": 1},
    # No CWE → filtered out
    {"func_before": "void vuln4() {}", "CWE ID": None,
     "commit_id": "ccc", "CVSS Score": 5.0, "target": 1},
    # Single-function commit, low CVSS → filtered when threshold set
    {"func_before": "void low_cvss() {}", "CWE ID": "CWE-476",
     "commit_id": "ddd", "CVSS Score": 3.1, "target": 1},
    # Single-function commit, null CVSS → filtered when threshold set
    {"func_before": "void null_cvss() {}", "CWE ID": "CWE-476",
     "commit_id": "eee", "CVSS Score": None, "target": 1},
]


@pytest.fixture
def megavul_path(tmp_path):
    p = tmp_path / "megavul.json"
    p.write_text(json.dumps(FIXTURE))
    return p


def test_extracts_valid_positive(megavul_path):
    samples = extract_megavul(megavul_path)
    assert any(s.code == "void vuln() {}" for s in samples)


def test_filters_multi_function_commit(megavul_path):
    samples = extract_megavul(megavul_path)
    codes = [s.code for s in samples]
    assert "void vuln2() {}" not in codes
    assert "void vuln3() {}" not in codes


def test_filters_missing_cwe(megavul_path):
    samples = extract_megavul(megavul_path)
    codes = [s.code for s in samples]
    assert "void vuln4() {}" not in codes


def test_no_cvss_filter_by_default(megavul_path):
    samples = extract_megavul(megavul_path)
    codes = [s.code for s in samples]
    assert "void low_cvss() {}" in codes
    assert "void null_cvss() {}" in codes


def test_cvss_threshold_filters_low_and_null(megavul_path):
    samples = extract_megavul(megavul_path, cvss_threshold=7.0)
    codes = [s.code for s in samples]
    assert "void low_cvss() {}" not in codes
    assert "void null_cvss() {}" not in codes
    assert "void vuln() {}" in codes


def test_all_label_1(megavul_path):
    samples = extract_megavul(megavul_path)
    assert all(s.label == 1 for s in samples)

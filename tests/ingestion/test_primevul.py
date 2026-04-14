import json
import pytest
from pathlib import Path
from ingestion.primevul import extract_primevul
from ingestion.schema import FunctionSample


FIXTURE = [
    # commit "aaa": single target=1, CWE present → valid positive
    {"func": "void vuln() { return buf[i]; }", "cwe": "CWE-119", "target": 1,
     "commit_id": "aaa", "project": "proj"},
    # commit "aaa": target=0 → valid negative
    {"func": "void safe() { return 0; }", "cwe": "", "target": 0,
     "commit_id": "aaa", "project": "proj"},
    # commit "bbb": two target=1 entries → both filtered out (multi-function commit)
    {"func": "void vuln2() {}", "cwe": "CWE-787", "target": 1,
     "commit_id": "bbb", "project": "proj"},
    {"func": "void vuln3() {}", "cwe": "CWE-119", "target": 1,
     "commit_id": "bbb", "project": "proj"},
    # commit "ccc": single target=1 but no CWE → filtered out
    {"func": "void vuln4() {}", "cwe": "", "target": 1,
     "commit_id": "ccc", "project": "proj"},
    # commit "ddd": single target=1, multi-CWE string → valid positive with 2 CWEs
    {"func": "void vuln5() {}", "cwe": "['CWE-119', 'CWE-787']", "target": 1,
     "commit_id": "ddd", "project": "proj"},
]


@pytest.fixture
def primevul_path(tmp_path):
    p = tmp_path / "primevul.json"
    p.write_text(json.dumps(FIXTURE))
    return p


def test_extracts_single_function_commit_positive(primevul_path):
    samples = extract_primevul(primevul_path)
    positives = [s for s in samples if s.label == 1]
    codes = [s.code for s in positives]
    assert "void vuln() { return buf[i]; }" in codes


def test_filters_multi_function_commit_positives(primevul_path):
    samples = extract_primevul(primevul_path)
    codes = [s.code for s in samples]
    assert "void vuln2() {}" not in codes
    assert "void vuln3() {}" not in codes


def test_filters_positive_with_no_cwe(primevul_path):
    samples = extract_primevul(primevul_path)
    codes = [s.code for s in samples]
    assert "void vuln4() {}" not in codes


def test_keeps_all_negatives(primevul_path):
    samples = extract_primevul(primevul_path)
    negatives = [s for s in samples if s.label == 0]
    assert any(s.code == "void safe() { return 0; }" for s in negatives)
    assert all(s.cwes == [] for s in negatives)


def test_parses_multi_cwe_string(primevul_path):
    samples = extract_primevul(primevul_path)
    multi = next(s for s in samples if s.code == "void vuln5() {}")
    assert set(multi.cwes) == {"CWE-119", "CWE-787"}


def test_returns_function_samples(primevul_path):
    samples = extract_primevul(primevul_path)
    assert all(isinstance(s, FunctionSample) for s in samples)

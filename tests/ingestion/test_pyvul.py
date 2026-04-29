import json
import pytest
from ingestion.pyvul import extract_pyvul
from ingestion.schema import FunctionSample


@pytest.fixture
def pyvul_json_path(tmp_path):
    records = [
        {"code": "def bad():\n    pass", "cwe": "CWE-089", "label": 1},
        {"code": "def safe():\n    pass", "cwe": "",        "label": 0},
        {"code": "def nvd():\n    pass",  "cwe": "NVD-CWE-noinfo", "label": 1},
        {"code": "",                       "cwe": "CWE-22", "label": 1},
    ]
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(records))
    return tmp_path


def test_extracts_positive(pyvul_json_path):
    samples = extract_pyvul(pyvul_json_path)
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 1
    assert positives[0].cwes == ["CWE-89"]


def test_normalises_cwe(pyvul_json_path):
    samples = extract_pyvul(pyvul_json_path)
    positives = [s for s in samples if s.label == 1]
    assert all(not c.startswith("CWE-0") for s in positives for c in s.cwes)


def test_extracts_negative(pyvul_json_path):
    samples = extract_pyvul(pyvul_json_path)
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1


def test_drops_nvd_placeholder(pyvul_json_path):
    samples = extract_pyvul(pyvul_json_path)
    codes = [s.code for s in samples]
    assert "def nvd():\n    pass" not in codes


def test_branch_and_language(pyvul_json_path):
    samples = extract_pyvul(pyvul_json_path)
    assert all(s.branch == "real" for s in samples)
    assert all(s.language == "Python" for s in samples)

import json
import pytest
from ingestion.vul4j import extract_vul4j
from ingestion.schema import FunctionSample


@pytest.fixture
def vul4j_path(tmp_path):
    records = [
        {"cwe": "CWE-022", "vulnerable_code": "public void trav(){}", "fixed_code": "public void safe(){}"},
        {"cwe": "",         "vulnerable_code": "public void x(){}",    "fixed_code": ""},
    ]
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(records))
    return tmp_path


def test_extracts_positive(vul4j_path):
    samples = extract_vul4j(vul4j_path)
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 1
    assert positives[0].cwes == ["CWE-22"]


def test_extracts_negative(vul4j_path):
    samples = extract_vul4j(vul4j_path)
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1
    assert negatives[0].cwes == []


def test_drops_empty_cwe(vul4j_path):
    samples = extract_vul4j(vul4j_path)
    positives = [s for s in samples if s.label == 1]
    assert all(s.cwes for s in positives)


def test_branch_and_language(vul4j_path):
    samples = extract_vul4j(vul4j_path)
    assert all(s.branch == "real" for s in samples)
    assert all(s.language == "Java" for s in samples)

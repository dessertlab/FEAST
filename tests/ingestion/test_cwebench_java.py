import json
import pytest
from ingestion.cwebench_java import extract_cwebench_java
from ingestion.schema import FunctionSample


@pytest.fixture
def cwebench_path(tmp_path):
    records = [
        {"cwe": "CWE-022", "vulnerable_code": "public void bad(){trav();}", "fixed_code": "public void fixed(){}"},
        {"cwe": "CWE-078", "vulnerable_code": "public void cmd(){exec(input);}", "fixed_code": ""},
        {"cwe": "",         "vulnerable_code": "public void x(){}",              "fixed_code": ""},
    ]
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(records))
    return tmp_path


def test_normalises_cwe(cwebench_path):
    samples = extract_cwebench_java(cwebench_path)
    positives = [s for s in samples if s.label == 1]
    cwes = {s.cwes[0] for s in positives if s.cwes}
    assert "CWE-22" in cwes   # CWE-022 -> CWE-22


def test_extracts_fixed_as_negative(cwebench_path):
    samples = extract_cwebench_java(cwebench_path)
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1
    assert negatives[0].cwes == []


def test_drops_empty_cwe(cwebench_path):
    samples = extract_cwebench_java(cwebench_path)
    positives = [s for s in samples if s.label == 1]
    assert all(s.cwes for s in positives)


def test_branch_and_language(cwebench_path):
    samples = extract_cwebench_java(cwebench_path)
    assert all(s.branch == "real" for s in samples)
    assert all(s.language == "Java" for s in samples)

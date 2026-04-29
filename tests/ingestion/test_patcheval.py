import json
import pytest
from ingestion.patcheval import extract_patcheval
from ingestion.schema import FunctionSample


@pytest.fixture
def patcheval_path(tmp_path):
    records = [
        {
            "cve_id": "CVE-2021-1",
            "cwe_info": {"CWE-22": "Path Traversal"},
            "vul_func": "def bad():\n    open(path)",
            "fix_func":  "def fixed():\n    open(safe_path)",
            "programming_language": "Python",
        },
        {
            "cve_id": "CVE-2021-2",
            "cwe_info": {"CWE-78": "Command Injection"},
            "vul_func": "void cmd(){exec(input);}",
            "fix_func":  "",
            "programming_language": "C",
        },
        {
            "cve_id": "CVE-2021-3",
            "cwe_info": {},
            "vul_func": "def no_cwe(): pass",
            "fix_func":  "",
            "programming_language": "Python",
        },
    ]
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(records))
    return p


def test_filters_python_only(patcheval_path):
    samples = extract_patcheval(patcheval_path)
    assert all(s.language == "Python" for s in samples)


def test_extracts_positive(patcheval_path):
    samples = extract_patcheval(patcheval_path)
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 1
    assert positives[0].cwes == ["CWE-22"]


def test_extracts_negative(patcheval_path):
    samples = extract_patcheval(patcheval_path)
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1
    assert negatives[0].cwes == []


def test_drops_empty_cwe_info(patcheval_path):
    samples = extract_patcheval(patcheval_path)
    codes = [s.code for s in samples]
    assert "def no_cwe(): pass" not in codes


def test_branch_and_language(patcheval_path):
    samples = extract_patcheval(patcheval_path)
    assert all(s.branch == "real" for s in samples)

import pandas as pd
import pytest
from ingestion.synth_vuln_fixes import extract_synth_vuln_fixes
from ingestion.schema import FunctionSample


@pytest.fixture
def svf_path(tmp_path):
    df = pd.DataFrame([
        {"cwe": "CWE-22",  "vulnerable_code": "def bad():\n    open(path)", "fixed_code": "def ok():\n    open(safe)", "language": "python"},
        {"cwe": "CWE-89",  "vulnerable_code": "def sql(): pass",            "fixed_code": "",                          "language": "python"},
        {"cwe": "NVD-CWE-noinfo", "vulnerable_code": "def nvd(): pass",    "fixed_code": "",                          "language": "python"},
        {"cwe": "CWE-79",  "vulnerable_code": "public class X {}",         "fixed_code": "",                          "language": "java"},
    ])
    p = tmp_path / "data.parquet"
    df.to_parquet(p)
    return p


def test_extracts_positives(svf_path):
    samples = extract_synth_vuln_fixes(svf_path)
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 2  # CWE-22 and CWE-89 Python rows


def test_extracts_negative(svf_path):
    samples = extract_synth_vuln_fixes(svf_path)
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1
    assert negatives[0].cwes == []


def test_drops_nvd_placeholder(svf_path):
    samples = extract_synth_vuln_fixes(svf_path)
    codes = [s.code for s in samples]
    assert "def nvd(): pass" not in codes


def test_branch_and_language(svf_path):
    samples = extract_synth_vuln_fixes(svf_path)
    assert all(s.branch == "ai" for s in samples)
    assert all(s.language == "Python" for s in samples)

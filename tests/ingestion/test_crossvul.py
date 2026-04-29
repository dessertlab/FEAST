import pandas as pd
import pytest
from pathlib import Path
from ingestion.crossvul import extract_crossvul
from ingestion.schema import FunctionSample


@pytest.fixture
def crossvul_path(tmp_path):
    df = pd.DataFrame([
        {"language": "c",      "cwe_id": "CWE-119", "vulnerable_code": "void v(){}", "fixed_code": "void f(){}"},
        {"language": "java",   "cwe_id": "CWE-89",  "vulnerable_code": "void v(){}", "fixed_code": "void f(){}"},
        {"language": "c",      "cwe_id": "",         "vulnerable_code": "void v(){}", "fixed_code": "void f(){}"},
        {"language": "c",      "cwe_id": "NVD-CWE-noinfo", "vulnerable_code": "v", "fixed_code": "f"},
        {"language": "python", "cwe_id": "CWE-22",  "vulnerable_code": "x=1", "fixed_code": "y=2"},
    ])
    p = tmp_path / "crossvul.parquet"
    df.to_parquet(p)
    return p


def test_returns_only_label0(crossvul_path):
    samples = extract_crossvul(crossvul_path, language="C/C++")
    assert all(s.label == 0 for s in samples)


def test_filters_by_language(crossvul_path):
    c_samples  = extract_crossvul(crossvul_path, language="C/C++")
    java_samples = extract_crossvul(crossvul_path, language="Java")
    assert len(c_samples) == 1
    assert len(java_samples) == 1


def test_drops_empty_cwe(crossvul_path):
    samples = extract_crossvul(crossvul_path, language="C/C++")
    # row with empty cwe_id should be filtered out; only 1 C row with valid CWE remains
    assert len(samples) == 1


def test_branch_and_language_fields(crossvul_path):
    samples = extract_crossvul(crossvul_path, language="C/C++")
    assert all(s.branch == "real" for s in samples)
    assert all(s.language == "C/C++" for s in samples)


def test_returns_function_samples(crossvul_path):
    samples = extract_crossvul(crossvul_path, language="C/C++")
    assert all(isinstance(s, FunctionSample) for s in samples)

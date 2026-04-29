import pytest
from ingestion.formai import extract_formai
from ingestion.schema import FunctionSample
import pandas as pd


@pytest.fixture
def formai_path(tmp_path):
    df = pd.DataFrame([
        {"filename": "test.c",   "classification": "Vulnerable",     "source_code": "void v(){buf[100]=1;}", "vul_type": "buffer overflow"},
        {"filename": "test.c",   "classification": "Non-Vulnerable",  "source_code": "void s(){buf[0]=1;}",  "vul_type": ""},
        {"filename": "test.c",   "classification": "Vulnerable",      "source_code": "void x(){}",           "vul_type": "unknown_type_xyz"},
        {"filename": "test.py",  "classification": "Vulnerable",      "source_code": "def bad():",           "vul_type": "buffer overflow"},
    ])
    p = tmp_path / "formai.csv"
    df.to_csv(p, index=False)
    return p


def test_extracts_positive(formai_path):
    samples = extract_formai(formai_path)
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 1
    assert positives[0].cwes == ["CWE-121"]


def test_extracts_negative(formai_path):
    samples = extract_formai(formai_path)
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1


def test_drops_unknown_vul_type(formai_path):
    samples = extract_formai(formai_path)
    codes = [s.code for s in samples]
    assert "void x(){}" not in codes


def test_skips_non_c_files(formai_path):
    samples = extract_formai(formai_path)
    codes = [s.code for s in samples]
    assert "def bad():" not in codes


def test_branch_and_language(formai_path):
    samples = extract_formai(formai_path)
    assert all(s.branch == "ai" for s in samples)
    assert all(s.language == "C/C++" for s in samples)

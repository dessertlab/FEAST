import pandas as pd
import pytest
from ingestion.sven import extract_sven
from ingestion.schema import FunctionSample


@pytest.fixture
def sven_path(tmp_path):
    df = pd.DataFrame([
        {"file_name": "test.c",  "func_src_before": "void v(){}", "func_src_after": "void f(){}", "vul_type": "cwe-119"},
        {"file_name": "test.py", "func_src_before": "def v():",   "func_src_after": "def f():",   "vul_type": "cwe-089"},
        {"file_name": "test.c",  "func_src_before": "void x(){}", "func_src_after": "void y(){}", "vul_type": ""},
        {"file_name": "test.c",  "func_src_before": "void z(){}", "func_src_after": "",            "vul_type": "cwe-022"},
    ])
    p = tmp_path / "sven.parquet"
    df.to_parquet(p)
    return p


def test_normalises_cwe(sven_path):
    samples = extract_sven(sven_path, language="C/C++")
    positives = [s for s in samples if s.label == 1]
    assert any(s.cwes == ["CWE-119"] for s in positives)


def test_filters_language(sven_path):
    c_samples  = extract_sven(sven_path, language="C/C++")
    py_samples = extract_sven(sven_path, language="Python")
    codes_c  = {s.code for s in c_samples}
    codes_py = {s.code for s in py_samples}
    assert "def v():" not in codes_c
    assert "void v(){}" not in codes_py


def test_drops_empty_vul_type(sven_path):
    samples = extract_sven(sven_path, language="C/C++")
    codes = [s.code for s in samples]
    assert "void x(){}" not in codes


def test_branch_and_language(sven_path):
    samples = extract_sven(sven_path, language="C/C++")
    assert all(s.branch == "real" for s in samples)
    assert all(s.language == "C/C++" for s in samples)


def test_safe_samples_have_empty_cwes(sven_path):
    samples = extract_sven(sven_path, language="C/C++")
    negatives = [s for s in samples if s.label == 0]
    assert all(s.cwes == [] for s in negatives)

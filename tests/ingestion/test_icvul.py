import csv
import pytest
from pathlib import Path
from ingestion.icvul import extract_icvul
from ingestion.schema import FunctionSample


@pytest.fixture
def icvul_path(tmp_path):
    # functions.csv
    functions = tmp_path / "functions.csv"
    functions.write_text(
        "func_id,func_before\n"
        "f1,void vuln_vcc() { return buf[i]; }\n"
        "f2,void not_vcc() { return 0; }\n"
        'f3,"void vuln_vcc2() { strcpy(dst, src); }"\n'
    )
    # cve_fc_vcc_mapping.csv
    mapping = tmp_path / "cve_fc_vcc_mapping.csv"
    mapping.write_text(
        "cwe_id,vcc_func_id\n"
        "CWE-119,f1\n"
        "CWE-120,f3\n"
        # f2 has no entry → not a VCC
    )
    return tmp_path


def test_extracts_only_vcc_functions(icvul_path):
    samples = extract_icvul(icvul_path)
    codes = [s.code for s in samples]
    assert "void vuln_vcc() { return buf[i]; }" in codes
    assert "void vuln_vcc2() { strcpy(dst, src); }" in codes
    assert "void not_vcc() { return 0; }" not in codes


def test_no_negatives_produced(icvul_path):
    samples = extract_icvul(icvul_path)
    assert all(s.label == 1 for s in samples)


def test_cwe_assigned_to_vcc_function(icvul_path):
    samples = extract_icvul(icvul_path)
    f1 = next(s for s in samples if s.code == "void vuln_vcc() { return buf[i]; }")
    assert f1.cwes == ["CWE-119"]


def test_returns_function_samples(icvul_path):
    samples = extract_icvul(icvul_path)
    assert all(isinstance(s, FunctionSample) for s in samples)

# tests/ingestion/test_schema.py
from ingestion.schema import FunctionSample


def test_positive_sample():
    s = FunctionSample(code="int foo() { return buf[i]; }", cwes=["CWE-119"], label=1)
    assert s.code == "int foo() { return buf[i]; }"
    assert s.cwes == ["CWE-119"]
    assert s.label == 1


def test_negative_sample_has_empty_cwes():
    s = FunctionSample(code="int bar() { return 0; }", cwes=[], label=0)
    assert s.cwes == []
    assert s.label == 0


def test_multi_cwe_sample():
    s = FunctionSample(code="void baz() {}", cwes=["CWE-119", "CWE-787"], label=1)
    assert len(s.cwes) == 2
    assert "CWE-787" in s.cwes

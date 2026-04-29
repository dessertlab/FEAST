import json
import pytest
from ingestion.security_eval import extract_security_eval
from ingestion.schema import FunctionSample


@pytest.fixture
def security_eval_jsonl(tmp_path):
    lines = [
        {"ID": "CWE-79-1",  "Insecure_code": "print(user_input)",   "CWE": "CWE-79"},
        {"ID": "CWE-22-1",  "Insecure_code": "open(path)",          "CWE": "CWE-22"},
        {"ID": "NO_CWE",    "Insecure_code": "x = 1",               "CWE": ""},
        {"ID": "EMPTY",     "Insecure_code": "",                     "CWE": "CWE-89"},
    ]
    p = tmp_path / "dataset.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in lines))
    return p


@pytest.fixture
def security_eval_dirs(tmp_path):
    cwe79 = tmp_path / "CWE-79"
    cwe79.mkdir()
    (cwe79 / "xss.py").write_text("print(user_input)")
    return tmp_path


def test_jsonl_extracts_positives(security_eval_jsonl):
    samples = extract_security_eval(security_eval_jsonl)
    assert len(samples) == 2
    assert all(s.label == 1 for s in samples)


def test_jsonl_drops_empty_cwe(security_eval_jsonl):
    samples = extract_security_eval(security_eval_jsonl)
    codes = [s.code for s in samples]
    assert "x = 1" not in codes


def test_jsonl_drops_empty_code(security_eval_jsonl):
    samples = extract_security_eval(security_eval_jsonl)
    assert all(s.code for s in samples)


def test_dir_layout(security_eval_dirs):
    samples = extract_security_eval(security_eval_dirs)
    assert len(samples) == 1
    assert samples[0].cwes == ["CWE-79"]


def test_branch_and_language(security_eval_jsonl):
    samples = extract_security_eval(security_eval_jsonl)
    assert all(s.branch == "ai" for s in samples)
    assert all(s.language == "Python" for s in samples)

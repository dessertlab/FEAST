import pytest
from ingestion.llmseceval import extract_llmseceval
from ingestion.schema import FunctionSample


@pytest.fixture
def llmseceval_dir(tmp_path):
    cwe121 = tmp_path / "CWE-121"
    cwe121.mkdir()
    (cwe121 / "exploit.c").write_text("void vuln(){buf[200]=1;}")
    secure = cwe121 / "Secure"
    secure.mkdir()
    (secure / "safe.c").write_text("void safe(){buf[0]=1;}")

    cwe89 = tmp_path / "CWE-89"
    cwe89.mkdir()
    (cwe89 / "inject.c").write_text("void sql(){query(input);}")

    # Python file in C dir should be ignored when language=C/C++
    (cwe121 / "script.py").write_text("import os")
    return tmp_path


def test_extracts_positives(llmseceval_dir):
    samples = extract_llmseceval(llmseceval_dir, language="C/C++")
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 2


def test_extracts_negatives(llmseceval_dir):
    samples = extract_llmseceval(llmseceval_dir, language="C/C++")
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1
    assert negatives[0].cwes == []


def test_cwe_from_dir(llmseceval_dir):
    samples = extract_llmseceval(llmseceval_dir, language="C/C++")
    pos_121 = [s for s in samples if s.label == 1 and s.cwes == ["CWE-121"]]
    assert len(pos_121) == 1


def test_branch_and_language(llmseceval_dir):
    samples = extract_llmseceval(llmseceval_dir, language="C/C++")
    assert all(s.branch == "ai" for s in samples)
    assert all(s.language == "C/C++" for s in samples)


def test_language_filter_excludes_wrong_ext(llmseceval_dir):
    samples = extract_llmseceval(llmseceval_dir, language="C/C++")
    codes = [s.code for s in samples]
    assert "import os" not in codes

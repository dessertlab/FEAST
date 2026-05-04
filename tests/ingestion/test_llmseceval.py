import pytest
from pathlib import Path
from ingestion.llmseceval import extract_llmseceval
from ingestion.schema import FunctionSample


@pytest.fixture
def llmseceval_dir(tmp_path):
    # Vulnerable C/C++ via zenodo gen_scenario structure
    gen_c121 = tmp_path / "zenodo" / "data" / "cwe-121" / "scenario1" / "gen_scenario"
    gen_c121.mkdir(parents=True)
    (gen_c121 / "test_copilot_1.c").write_text("void vuln(){buf[200]=1;}")

    gen_c89 = tmp_path / "zenodo" / "data" / "cwe-89" / "scenario1" / "gen_scenario"
    gen_c89.mkdir(parents=True)
    (gen_c89 / "test_copilot_1.c").write_text("void sql(){query(input);}")

    # Python file in C gen_scenario -- ignored for C/C++
    (gen_c121 / "script.py").write_text("import os")

    # Safe Python via CWE-NNN/Secure structure
    secure_dir = tmp_path / "CWE-121" / "Secure"
    secure_dir.mkdir(parents=True)
    (secure_dir / "safe.py").write_text("def safe(): pass")

    return tmp_path


def test_extracts_positives(llmseceval_dir):
    samples = extract_llmseceval(llmseceval_dir, language="C/C++")
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 2


def test_no_negatives_for_c(llmseceval_dir):
    # extractor only collects safe samples for Python, not C/C++
    samples = extract_llmseceval(llmseceval_dir, language="C/C++")
    assert all(s.label == 1 for s in samples)


def test_extracts_python_negatives(llmseceval_dir):
    samples = extract_llmseceval(llmseceval_dir, language="Python")
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

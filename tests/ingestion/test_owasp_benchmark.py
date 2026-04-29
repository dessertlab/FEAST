import pytest
from ingestion.owasp_benchmark import extract_owasp_benchmark
from ingestion.schema import FunctionSample


@pytest.fixture
def owasp_java_dir(tmp_path):
    csv_content = (
        "# test name,category,real vulnerability,cwe\n"
        "BenchmarkTest00001,sqli,1,CWE-89\n"
        "BenchmarkTest00002,xss,0,CWE-79\n"
        "BenchmarkTest00003,cmdi,1,CWE-78\n"
    )
    (tmp_path / "expectedresults-1.2.csv").write_text(csv_content)
    src = tmp_path / "src" / "main" / "java"
    src.mkdir(parents=True)
    (src / "BenchmarkTest00001.java").write_text("public class BenchmarkTest00001 { }")
    (src / "BenchmarkTest00002.java").write_text("public class BenchmarkTest00002 { }")
    (src / "BenchmarkTest00003.java").write_text("public class BenchmarkTest00003 { }")
    return tmp_path


def test_extracts_positives(owasp_java_dir):
    samples = extract_owasp_benchmark(owasp_java_dir, language="Java")
    positives = [s for s in samples if s.label == 1]
    assert len(positives) == 2


def test_extracts_negatives(owasp_java_dir):
    samples = extract_owasp_benchmark(owasp_java_dir, language="Java")
    negatives = [s for s in samples if s.label == 0]
    assert len(negatives) == 1
    assert negatives[0].cwes == []


def test_cwe_from_csv(owasp_java_dir):
    samples = extract_owasp_benchmark(owasp_java_dir, language="Java")
    pos_sql = [s for s in samples if s.label == 1 and "CWE-89" in s.cwes]
    assert len(pos_sql) == 1


def test_branch_and_language(owasp_java_dir):
    samples = extract_owasp_benchmark(owasp_java_dir, language="Java")
    assert all(s.branch == "synth" for s in samples)
    assert all(s.language == "Java" for s in samples)

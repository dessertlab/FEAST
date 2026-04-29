import json
import pytest
from ingestion.capec_llm import extract_capec_llm
from ingestion.schema import FunctionSample


@pytest.fixture
def capec_java_path(tmp_path):
    records = [
        {
            "capec_id": "CAPEC-1",
            "code_snippet": "public class Vuln { public void exec(String input) { Runtime.getRuntime().exec(input); } }",
            "description": "This maps to CWE-78 command injection.",
        },
        {
            "capec_id": "CAPEC-2",
            "code_snippet": "def run(cmd): import os; os.system(cmd)",
            "description": "Maps to CWE-78.",
        },
        {
            "capec_id": "CAPEC-3",
            "code_snippet": "public class X { void x(){} }",
            "description": "No CWE mentioned here.",
        },
    ]
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(records))
    return tmp_path


def test_extracts_java_only(capec_java_path):
    samples = extract_capec_llm(capec_java_path, language="Java")
    assert all(s.language == "Java" for s in samples)
    codes = [s.code for s in samples]
    assert not any("def run" in c for c in codes)


def test_drops_no_cwe(capec_java_path):
    samples = extract_capec_llm(capec_java_path, language="Java")
    # CAPEC-3 has no CWE in description -> dropped
    codes = [s.code for s in samples]
    assert not any("void x(){}" in c for c in codes)


def test_cwe_from_description(capec_java_path):
    samples = extract_capec_llm(capec_java_path, language="Java")
    assert all("CWE-78" in s.cwes for s in samples)


def test_branch_and_language(capec_java_path):
    samples = extract_capec_llm(capec_java_path, language="Java")
    assert all(s.branch == "ai" for s in samples)

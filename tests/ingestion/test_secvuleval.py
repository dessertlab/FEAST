import csv
import json
import pytest
from pathlib import Path
from ingestion.secvuleval import extract_secvuleval
from ingestion.schema import FunctionSample


NVD_CACHE = {
    "CVE-2021-001": ["CWE-119"],
    "CVE-2021-002": ["CWE-787", "CWE-119"],
    # CVE-2021-003 intentionally absent (unresolvable)
}

FIXTURE_ROWS = [
    # Positive, CWE resolvable → valid
    {"func": "void vuln1() {}", "label": "1", "cve_id": "CVE-2021-001"},
    # Positive, multi-CWE → valid with 2 CWEs
    {"func": "void vuln2() {}", "label": "1", "cve_id": "CVE-2021-002"},
    # Positive, CWE unresolvable → filtered out
    {"func": "void vuln3() {}", "label": "1", "cve_id": "CVE-2021-003"},
    # Negative → kept as-is, no CWE lookup
    {"func": "void safe1() {}", "label": "0", "cve_id": "CVE-2021-001"},
    {"func": "void safe2() {}", "label": "0", "cve_id": "CVE-2021-003"},
]


@pytest.fixture
def secvuleval_files(tmp_path):
    data_path = tmp_path / "secvuleval.csv"
    with open(data_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["func", "label", "cve_id"])
        writer.writeheader()
        writer.writerows(FIXTURE_ROWS)

    cache_path = tmp_path / "nvd_cache.json"
    cache_path.write_text(json.dumps(NVD_CACHE))

    return data_path, cache_path


def test_extracts_resolvable_positive(secvuleval_files):
    data_path, cache_path = secvuleval_files
    samples = extract_secvuleval(data_path, cache_path)
    codes = [s.code for s in samples]
    assert "void vuln1() {}" in codes


def test_assigns_cwe_from_nvd_cache(secvuleval_files):
    data_path, cache_path = secvuleval_files
    samples = extract_secvuleval(data_path, cache_path)
    vuln1 = next(s for s in samples if s.code == "void vuln1() {}")
    assert vuln1.cwes == ["CWE-119"]


def test_assigns_multi_cwe(secvuleval_files):
    data_path, cache_path = secvuleval_files
    samples = extract_secvuleval(data_path, cache_path)
    vuln2 = next(s for s in samples if s.code == "void vuln2() {}")
    assert set(vuln2.cwes) == {"CWE-787", "CWE-119"}


def test_filters_unresolvable_positive(secvuleval_files):
    data_path, cache_path = secvuleval_files
    samples = extract_secvuleval(data_path, cache_path)
    codes = [s.code for s in samples]
    assert "void vuln3() {}" not in codes


def test_keeps_all_negatives(secvuleval_files):
    data_path, cache_path = secvuleval_files
    samples = extract_secvuleval(data_path, cache_path)
    negatives = [s for s in samples if s.label == 0]
    neg_codes = [s.code for s in negatives]
    assert "void safe1() {}" in neg_codes
    assert "void safe2() {}" in neg_codes


def test_negatives_have_empty_cwes(secvuleval_files):
    data_path, cache_path = secvuleval_files
    samples = extract_secvuleval(data_path, cache_path)
    for s in samples:
        if s.label == 0:
            assert s.cwes == []


def test_returns_function_samples(secvuleval_files):
    data_path, cache_path = secvuleval_files
    samples = extract_secvuleval(data_path, cache_path)
    assert all(isinstance(s, FunctionSample) for s in samples)

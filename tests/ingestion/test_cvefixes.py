import sqlite3
import pytest
from pathlib import Path
from ingestion.cvefixes import extract_cvefixes
from ingestion.schema import FunctionSample


@pytest.fixture
def cvefixes_db(tmp_path):
    db_path = tmp_path / "CVEfixes.db"
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    cur.executescript("""
        CREATE TABLE cve (cve_id TEXT PRIMARY KEY, cwe_id TEXT);
        CREATE TABLE fixes (cve_id TEXT, hash TEXT);
        CREATE TABLE commits (hash TEXT PRIMARY KEY);
        CREATE TABLE file_change (id INTEGER PRIMARY KEY, hash TEXT, filename TEXT);
        CREATE TABLE method_change (
            id INTEGER PRIMARY KEY,
            file_change_id INTEGER,
            code_before TEXT,
            code_after TEXT
        );

        -- CVE with CWE, one function changed → valid positive
        INSERT INTO cve VALUES ('CVE-2021-001', 'CWE-119');
        INSERT INTO fixes VALUES ('CVE-2021-001', 'hash_a');
        INSERT INTO commits VALUES ('hash_a');
        INSERT INTO file_change VALUES (1, 'hash_a', 'src/foo.c');
        INSERT INTO method_change VALUES (1, 1, 'void vuln() {}', 'void fixed() {}');

        -- CVE with CWE, two functions changed → filtered out (multi-function commit)
        INSERT INTO cve VALUES ('CVE-2021-002', 'CWE-787');
        INSERT INTO fixes VALUES ('CVE-2021-002', 'hash_b');
        INSERT INTO commits VALUES ('hash_b');
        INSERT INTO file_change VALUES (2, 'hash_b', 'src/bar.c');
        INSERT INTO method_change VALUES (2, 2, 'void multi1() {}', 'void fixed1() {}');
        INSERT INTO method_change VALUES (3, 2, 'void multi2() {}', 'void fixed2() {}');

        -- CVE with no CWE → filtered out
        INSERT INTO cve VALUES ('CVE-2021-003', NULL);
        INSERT INTO fixes VALUES ('CVE-2021-003', 'hash_c');
        INSERT INTO commits VALUES ('hash_c');
        INSERT INTO file_change VALUES (3, 'hash_c', 'src/baz.c');
        INSERT INTO method_change VALUES (4, 3, 'void no_cwe() {}', 'void fixed3() {}');

        -- C++ file, valid CVE → valid positive
        INSERT INTO cve VALUES ('CVE-2021-004', 'CWE-476');
        INSERT INTO fixes VALUES ('CVE-2021-004', 'hash_d');
        INSERT INTO commits VALUES ('hash_d');
        INSERT INTO file_change VALUES (4, 'hash_d', 'src/qux.cpp');
        INSERT INTO method_change VALUES (5, 4, 'void cpp_vuln() {}', 'void cpp_fixed() {}');

        -- Non-C/C++ file → filtered out
        INSERT INTO cve VALUES ('CVE-2021-005', 'CWE-89');
        INSERT INTO fixes VALUES ('CVE-2021-005', 'hash_e');
        INSERT INTO commits VALUES ('hash_e');
        INSERT INTO file_change VALUES (5, 'hash_e', 'src/App.java');
        INSERT INTO method_change VALUES (6, 5, 'void java_method() {}', 'void fixed4() {}');
    """)
    con.commit()
    con.close()
    return db_path


def test_extracts_single_function_c_positive(cvefixes_db):
    samples = extract_cvefixes(cvefixes_db)
    codes = [s.code for s in samples]
    assert "void vuln() {}" in codes


def test_filters_multi_function_commit(cvefixes_db):
    samples = extract_cvefixes(cvefixes_db)
    codes = [s.code for s in samples]
    assert "void multi1() {}" not in codes
    assert "void multi2() {}" not in codes


def test_filters_missing_cwe(cvefixes_db):
    samples = extract_cvefixes(cvefixes_db)
    codes = [s.code for s in samples]
    assert "void no_cwe() {}" not in codes


def test_accepts_cpp_files(cvefixes_db):
    samples = extract_cvefixes(cvefixes_db)
    codes = [s.code for s in samples]
    assert "void cpp_vuln() {}" in codes


def test_filters_non_c_files(cvefixes_db):
    samples = extract_cvefixes(cvefixes_db)
    codes = [s.code for s in samples]
    assert "void java_method() {}" not in codes


def test_all_positives_label_1(cvefixes_db):
    samples = extract_cvefixes(cvefixes_db)
    assert all(s.label == 1 for s in samples)


def test_cwe_assigned(cvefixes_db):
    samples = extract_cvefixes(cvefixes_db)
    vuln = next(s for s in samples if s.code == "void vuln() {}")
    assert vuln.cwes == ["CWE-119"]

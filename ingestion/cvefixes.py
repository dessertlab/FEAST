import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ingestion.schema import FunctionSample

_C_EXTENSIONS = (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp")


@dataclass
class RawCVEfixesEntry:
    code_before: str
    cwes: list[str]
    label: int                   # always 1 (code_before is the vulnerable version)
    commit_hash: str
    num_functions_in_commit: int  # derived from DB; used for single-function filter


def _is_c_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in _C_EXTENSIONS)


def extract_cvefixes(db_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from a CVEfixes SQLite database.

    Positives: code_before entries where the file is C/C++, CWE is non-empty,
    and the commit modified exactly one function.

    Negatives: CVEfixes does not provide explicitly labeled safe samples beyond
    before/after pairs. code_after is not used as a negative (patched code is not
    ground-truth safe). This collection contributes positives only.
    """
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    # Count method changes per commit (for single-function filter)
    commit_func_counts = {
        row["hash"]: row["cnt"]
        for row in con.execute("""
            SELECT c.hash, COUNT(mc.id) AS cnt
            FROM commits c
            JOIN file_change fc ON fc.hash = c.hash
            JOIN method_change mc ON mc.file_change_id = fc.id
            WHERE mc.code_before IS NOT NULL AND mc.code_before != ''
            GROUP BY c.hash
        """)
    }

    rows = con.execute("""
        SELECT
            mc.code_before,
            cv.cwe_id,
            co.hash AS commit_hash,
            fc.filename
        FROM method_change mc
        JOIN file_change fc ON mc.file_change_id = fc.id
        JOIN commits co ON fc.hash = co.hash
        JOIN fixes f ON co.hash = f.hash
        JOIN cve cv ON f.cve_id = cv.cve_id
        WHERE mc.code_before IS NOT NULL
          AND mc.code_before != ''
          AND cv.cwe_id IS NOT NULL
          AND cv.cwe_id != ''
    """).fetchall()

    con.close()

    samples = []
    for row in rows:
        if not _is_c_file(row["filename"]):
            continue
        if commit_func_counts.get(row["commit_hash"], 0) != 1:
            continue
        samples.append(FunctionSample(
            code=row["code_before"],
            cwes=[row["cwe_id"].strip()],
            label=1,
        ))
    return samples

import csv
from pathlib import Path

from ingestion.schema import FunctionSample
from ingestion.utils import _CWE_RE

# OWASP Benchmark category -> CWE mapping
_CATEGORY_TO_CWE: dict[str, str] = {
    "cmdi":           "CWE-78",
    "commandinjection": "CWE-78",
    "ldapi":          "CWE-90",
    "pathtraver":     "CWE-22",
    "sqli":           "CWE-89",
    "sqlinjection":   "CWE-89",
    "securecookie":   "CWE-614",
    "trustbound":     "CWE-501",
    "weakrand":       "CWE-330",
    "weakcrypto":     "CWE-327",
    "xpathi":         "CWE-643",
    "xss":            "CWE-79",
    "crypto":         "CWE-327",
    "hash":           "CWE-328",
}


def _cwe_from_category(category: str) -> str | None:
    key = category.strip().lower()
    # Try direct map first
    if key in _CATEGORY_TO_CWE:
        return _CATEGORY_TO_CWE[key]
    # Try partial match
    for k, v in _CATEGORY_TO_CWE.items():
        if k in key:
            return v
    return None


def _normalise_cwe(raw: str) -> str:
    m = _CWE_RE.search(str(raw))
    if not m:
        return ""
    digits = m.group(0).replace("CWE-", "")
    try:
        return f"CWE-{int(digits)}"
    except ValueError:
        return ""


def extract_owasp_benchmark(
    data_path: Path,
    language: str = "Java",
) -> list[FunctionSample]:
    """Extract FunctionSamples from OWASP Benchmark.

    Source: OWASP GitHub.

    Java layout:
      data_path/
        expectedresults-1.2.csv    -- columns: # test name, category, real vulnerability, cwe
        src/main/java/.../testcode/
          BenchmarkTest00001.java

    Python layout (v0.1 preliminary):
      data_path/
        expectedresults.csv        -- columns: test name, category, real vulnerability, cwe
        testcode/
          BenchmarkTest00001.py

    language parameter accepts: "Java", "Python".
    """
    samples: list[FunctionSample] = []

    # Find the CSV results file
    csv_candidates = list(data_path.rglob("expectedresults*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No expectedresults*.csv found under {data_path}")
    csv_path = csv_candidates[0]

    # Read expected results
    expected: dict[str, tuple[bool, str]] = {}  # test_name -> (is_vuln, cwe)
    with open(csv_path, encoding="utf-8") as fh:
        all_lines = fh.readlines()
    # The first non-empty line is the header (may start with #); keep it.
    # Skip subsequent lines that are pure comments (start with # but no comma).
    header_found = False
    lines = []
    for line in all_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not header_found:
            header_found = True
            lines.append(line)
        elif stripped.startswith("#") and "," not in stripped:
            continue
        else:
            lines.append(line)
    reader = csv.DictReader(lines)
    for row in reader:
        # Column names vary; try both variants
        test_name = (
            row.get("# test name") or row.get("test name") or
            row.get("Test Name") or ""
        ).strip()
        category  = (row.get("category") or row.get("Category") or "").strip()
        real_vuln = (row.get("real vulnerability") or row.get("Real Vulnerability") or "0").strip()
        cwe_raw   = (row.get("cwe") or row.get("CWE") or "").strip()

        if not test_name:
            continue

        is_vuln = real_vuln in ("1", "true", "True", "TRUE", "yes")

        cwe = _normalise_cwe(cwe_raw) if cwe_raw else _cwe_from_category(category) or ""
        expected[test_name.lower()] = (is_vuln, cwe)

    # Find source files
    ext = ".java" if language == "Java" else ".py"
    src_files = list(data_path.rglob(f"*{ext}"))

    for fpath in sorted(src_files):
        stem = fpath.stem.lower()
        if stem not in expected:
            continue
        is_vuln, cwe = expected[stem]
        code = fpath.read_text(encoding="utf-8", errors="replace").strip()
        if not code:
            continue

        if is_vuln:
            if not cwe:
                continue
            samples.append(FunctionSample(
                code=code, cwes=[cwe], label=1,
                branch="synth", language=language,
            ))
        else:
            samples.append(FunctionSample(
                code=code, cwes=[], label=0,
                branch="synth", language=language,
            ))

    return samples

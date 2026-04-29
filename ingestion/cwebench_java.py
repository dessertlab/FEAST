import json
from pathlib import Path

from ingestion.schema import FunctionSample
from ingestion.utils import _CWE_RE

# Only these 4 CWEs are present in CWE-Bench-Java
_SUPPORTED_CWES = {"CWE-022", "CWE-078", "CWE-079", "CWE-094"}


def _normalise_cwe(raw: str) -> str:
    """'CWE-022' -> 'CWE-22'"""
    m = _CWE_RE.search(raw)
    if not m:
        return ""
    # re-parse to strip leading zeros
    digits = m.group(0).replace("CWE-", "")
    try:
        return f"CWE-{int(digits)}"
    except ValueError:
        return ""


def extract_cwebench_java(data_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from CWE-Bench-Java.

    Source: GitHub `iris-sast/cwe-bench-java`.

    Assumes code has been pre-extracted. Expected structure:
      data_path/
        project_info.csv   -- project metadata (optional, not used)
        fix_info.csv       -- optional, maps project to CWE
        <cwe>/
          <project>/
            vulnerable.java   -- label=1
            fixed.java        -- label=0 (optional)

    Alternatively, a flat JSON file `dataset.json` with records:
      {cwe: str, vulnerable_code: str, fixed_code: str}

    Both layouts are tried; whichever is found is used.
    """
    samples: list[FunctionSample] = []

    # Layout 1: flat JSON
    json_path = data_path / "dataset.json"
    if not json_path.exists():
        json_path = next(data_path.glob("*.json"), None)  # type: ignore[assignment]

    if json_path and json_path.exists():
        with open(json_path, encoding="utf-8") as fh:
            records = json.load(fh)
        for rec in records:
            cwe_raw = str(rec.get("cwe", "") or "")
            cwe = _normalise_cwe(cwe_raw)
            if not cwe:
                continue
            vuln = str(rec.get("vulnerable_code", "") or "").strip()
            if vuln:
                samples.append(FunctionSample(
                    code=vuln, cwes=[cwe], label=1,
                    branch="real", language="Java",
                ))
            fixed = str(rec.get("fixed_code", "") or "").strip()
            if fixed:
                samples.append(FunctionSample(
                    code=fixed, cwes=[], label=0,
                    branch="real", language="Java",
                ))
        return samples

    # Layout 2: directory tree  <cwe>/<project>/vulnerable.java
    for cwe_dir in sorted(data_path.iterdir()):
        if not cwe_dir.is_dir():
            continue
        cwe = _normalise_cwe(cwe_dir.name)
        if not cwe:
            continue
        for proj_dir in sorted(cwe_dir.iterdir()):
            if not proj_dir.is_dir():
                continue
            for fname, label, cwes in [
                ("vulnerable.java", 1, [cwe]),
                ("fixed.java",      0, []),
            ]:
                fpath = proj_dir / fname
                if not fpath.exists():
                    # try any .java file
                    candidates = list(proj_dir.glob("*.java"))
                    if not candidates:
                        continue
                    fpath = candidates[0]
                code = fpath.read_text(encoding="utf-8", errors="replace").strip()
                if not code:
                    continue
                samples.append(FunctionSample(
                    code=code, cwes=cwes, label=label,
                    branch="real", language="Java",
                ))

    return samples

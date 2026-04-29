import csv
from pathlib import Path

from ingestion.schema import FunctionSample
from ingestion.utils import _CWE_RE


def _normalise_cwe(raw: str) -> str:
    m = _CWE_RE.search(str(raw))
    if not m:
        return ""
    digits = m.group(0).replace("CWE-", "")
    try:
        return f"CWE-{int(digits)}"
    except ValueError:
        return ""


def extract_vul4j(data_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from Vul4J.

    Source: GitHub `tuhh-softsec/vul4j`.

    Assumes code has been pre-extracted by the authors' reproduction scripts.
    Expected structure:
      data_path/
        vul4j_dataset.csv    -- columns: vul_id, cve_id, cwe_id, ...
        extracted/
          <vul_id>/
            vulnerable/   -- .java files, label=1
            fixed/        -- .java files, label=0

    If vul4j_dataset.csv is not present the extractor tries a flat JSON layout
    (dataset.json) with records {cwe, vulnerable_code, fixed_code}.
    """
    samples: list[FunctionSample] = []

    # Flat JSON fallback (for testing / alternative distributions)
    json_path = data_path / "dataset.json"
    if not json_path.exists():
        json_path = next(data_path.glob("*.json"), None)  # type: ignore[assignment]
    if json_path and json_path.exists():
        import json
        with open(json_path, encoding="utf-8") as fh:
            records = json.load(fh)
        for rec in records:
            cwe = _normalise_cwe(str(rec.get("cwe", "")))
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

    # Standard layout: CSV + extracted/<vul_id>/vulnerable|fixed
    csv_path = data_path / "vul4j_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"vul4j_dataset.csv not found in {data_path}. "
            "Run the Vul4J reproduction scripts first."
        )

    cwe_by_id: dict[str, str] = {}
    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            vul_id = str(row.get("vul_id", "")).strip()
            cwe = _normalise_cwe(str(row.get("cwe_id", "")))
            if vul_id and cwe:
                cwe_by_id[vul_id] = cwe

    extracted_dir = data_path / "extracted"
    if not extracted_dir.exists():
        raise FileNotFoundError(
            f"extracted/ directory not found in {data_path}. "
            "Run the Vul4J reproduction scripts first."
        )

    for vul_dir in sorted(extracted_dir.iterdir()):
        if not vul_dir.is_dir():
            continue
        cwe = cwe_by_id.get(vul_dir.name)
        if not cwe:
            continue
        for subdir, label, cwes in [
            ("vulnerable", 1, [cwe]),
            ("fixed",      0, []),
        ]:
            src_dir = vul_dir / subdir
            if not src_dir.exists():
                continue
            for fpath in sorted(src_dir.rglob("*.java")):
                code = fpath.read_text(encoding="utf-8", errors="replace").strip()
                if not code:
                    continue
                samples.append(FunctionSample(
                    code=code, cwes=cwes, label=label,
                    branch="real", language="Java",
                ))

    return samples

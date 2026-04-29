import json
from pathlib import Path

from ingestion.schema import FunctionSample
from ingestion.utils import _CWE_RE, _NVD_PLACEHOLDERS


def _normalise_cwe(raw: str) -> str:
    m = _CWE_RE.search(str(raw))
    if not m:
        return ""
    digits = m.group(0).replace("CWE-", "")
    try:
        return f"CWE-{int(digits)}"
    except ValueError:
        return ""


def extract_pyvul(data_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from PyVul.

    Source: GitHub `billquan/PyVul`.

    Supports two layouts:
    1. Per-CWE JSONL files:  data_path/CWE-NNN.jsonl
       Each line: {code: str, label: int, ...}
    2. Single JSONL file:    data_path/dataset.jsonl
       Each line: {code: str, cwe: str, label: int, ...}
    3. JSON file:            data_path/dataset.json
       List of {code, cwe, label} records.

    Positives: label=1 with non-empty CWE.
    Negatives: label=0.
    """
    samples: list[FunctionSample] = []

    # Layout 3: single JSON file
    json_files = list(data_path.glob("*.json")) if data_path.is_dir() else [data_path]
    for jf in json_files:
        if not jf.exists() or jf.suffix != ".json":
            continue
        with open(jf, encoding="utf-8") as fh:
            records = json.load(fh)
        if not isinstance(records, list):
            continue
        cwe_from_name = _normalise_cwe(jf.stem)
        for rec in records:
            code = str(rec.get("code", "") or "").strip()
            if not code:
                continue
            label = int(rec.get("label", rec.get("vulnerable", 0)))
            cwe_raw = str(rec.get("cwe", "") or "")
            cwe = _normalise_cwe(cwe_raw) or cwe_from_name
            if label == 1:
                if not cwe or cwe in _NVD_PLACEHOLDERS:
                    continue
                samples.append(FunctionSample(
                    code=code, cwes=[cwe], label=1,
                    branch="real", language="Python",
                ))
            else:
                samples.append(FunctionSample(
                    code=code, cwes=[], label=0,
                    branch="real", language="Python",
                ))
        return samples

    # Layout 1 & 2: JSONL files
    if data_path.is_dir():
        jsonl_files = sorted(data_path.glob("*.jsonl"))
    elif data_path.suffix == ".jsonl":
        jsonl_files = [data_path]
    else:
        jsonl_files = []

    for jf in jsonl_files:
        cwe_from_name = _normalise_cwe(jf.stem)
        with open(jf, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                code = str(rec.get("code", "") or "").strip()
                if not code:
                    continue
                label = int(rec.get("label", rec.get("vulnerable", 0)))
                cwe_raw = str(rec.get("cwe", "") or "")
                cwe = _normalise_cwe(cwe_raw) or cwe_from_name
                if label == 1:
                    if not cwe or cwe in _NVD_PLACEHOLDERS:
                        continue
                    samples.append(FunctionSample(
                        code=code, cwes=[cwe], label=1,
                        branch="real", language="Python",
                    ))
                else:
                    samples.append(FunctionSample(
                        code=code, cwes=[], label=0,
                        branch="real", language="Python",
                    ))

    return samples

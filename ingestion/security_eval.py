import json
import re
from pathlib import Path

from ingestion.schema import FunctionSample
from ingestion.utils import _CWE_RE

_CWE_DIR_RE = re.compile(r"CWE[-_]?(\d+)", re.IGNORECASE)


def _cwe_from_id_field(raw) -> str:
    """Parse CWE from an ID field like 'CWE-79-1' or 'CWE79'."""
    if not raw:
        return ""
    m = _CWE_RE.search(str(raw))
    if m:
        digits = m.group(0).replace("CWE-", "")
        try:
            return f"CWE-{int(digits)}"
        except ValueError:
            return ""
    # Try without dash: 'CWE79'
    m2 = re.search(r"CWE(\d+)", str(raw), re.IGNORECASE)
    if m2:
        return f"CWE-{int(m2.group(1))}"
    return ""


def extract_security_eval(data_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from SecurityEval.

    Source: GitHub `s2e-lab/SecurityEval`.

    Supports two layouts:
    1. JSONL file (dataset.jsonl):
       Each line: {ID: str, Prompt: str, Insecure_code: str, CWE: str, ...}
    2. Directory of .py files organised by CWE:
       data_path/CWE-NNN/<file>.py  -- label=1 (all SecurityEval samples are vulnerable)

    SecurityEval only provides vulnerable samples (label=1); no safe samples.
    """
    samples: list[FunctionSample] = []

    # Layout 1: JSONL
    jsonl_candidates = list(data_path.glob("*.jsonl")) if data_path.is_dir() else []
    if data_path.suffix == ".jsonl":
        jsonl_candidates = [data_path]

    for jf in jsonl_candidates:
        with open(jf, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # Try multiple column name variants
                code = str(
                    rec.get("Insecure_code") or rec.get("insecure_code") or
                    rec.get("code") or rec.get("Code") or ""
                ).strip()
                if not code:
                    continue
                cwe_raw = str(
                    rec.get("CWE") or rec.get("cwe") or rec.get("ID") or ""
                )
                cwe = _cwe_from_id_field(cwe_raw)
                if not cwe:
                    continue
                samples.append(FunctionSample(
                    code=code, cwes=[cwe], label=1,
                    branch="ai", language="Python",
                ))
        return samples  # found JSONL, done

    # Layout 2: directory of .py files
    if data_path.is_dir():
        for cwe_dir in sorted(data_path.iterdir()):
            if not cwe_dir.is_dir():
                continue
            m = _CWE_DIR_RE.search(cwe_dir.name)
            if not m:
                continue
            cwe = f"CWE-{int(m.group(1))}"
            for fpath in sorted(cwe_dir.glob("*.py")):
                code = fpath.read_text(encoding="utf-8", errors="replace").strip()
                if not code:
                    continue
                samples.append(FunctionSample(
                    code=code, cwes=[cwe], label=1,
                    branch="ai", language="Python",
                ))

    return samples

import ast
from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample


def _parse_cwe_list(raw) -> list[str]:
    """Parse cwe_list field, which is a stringified Python list or NaN."""
    if raw is None or isinstance(raw, float):  # NaN
        return []
    raw = str(raw).strip()
    if not raw or raw in ("[]", "nan"):
        return []
    try:
        items = ast.literal_eval(raw)
        return [str(c).strip() for c in items if str(c).strip()]
    except (ValueError, SyntaxError):
        return []


def extract_secvuleval(data_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from a SecVulEval CSV file.

    Source: Hugging Face `arag0rn/SecVulEval`.
    Columns used: func_body, is_vulnerable, cwe_list.

    Positives: is_vulnerable=True entries with at least one CWE in cwe_list.
    Negatives: all is_vulnerable=False entries (cwes=[]).
    """
    df = pd.read_csv(data_path)

    samples = []
    for _, row in df.iterrows():
        is_vuln = bool(row["is_vulnerable"])
        code = str(row["func_body"])
        if is_vuln:
            cwes = _parse_cwe_list(row.get("cwe_list"))
            if cwes:
                samples.append(FunctionSample(code=code, cwes=cwes, label=1))
        else:
            samples.append(FunctionSample(code=code, cwes=[], label=0))
    return samples

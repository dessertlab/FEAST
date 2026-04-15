import ast
import re
from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample

_NVD_PLACEHOLDERS = {"NVD-CWE-Other", "NVD-CWE-noinfo", "NVD-CWE-Other "}
_CWE_RE = re.compile(r"CWE-\d+")


def _parse_cwe_list(raw) -> list[str]:
    """Parse cwe_list field, which is a stringified Python list or NaN.

    Handles:
    - NaN / None / empty         -> []
    - "['CWE-119']"              -> ["CWE-119"]
    - "['CWE-20CWE-190']"        -> ["CWE-20", "CWE-190"]  (concatenated, split by regex)
    - "['NVD-CWE-Other']"        -> []  (NVD placeholder, dropped)
    """
    if raw is None or isinstance(raw, float):  # NaN
        return []
    raw = str(raw).strip()
    if not raw or raw in ("[]", "nan"):
        return []
    try:
        items = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []

    result = []
    for item in items:
        item = str(item).strip()
        if not item or item in _NVD_PLACEHOLDERS:
            continue
        # Split concatenated CWE IDs (e.g. "CWE-20CWE-190" -> ["CWE-20", "CWE-190"])
        parts = _CWE_RE.findall(item)
        if parts:
            result.extend(parts)
        elif not item.startswith("CWE-"):
            # Non-CWE string (e.g. legacy label) -- skip
            pass
        else:
            result.append(item)
    return list(dict.fromkeys(result))  # deduplicate, preserve order


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

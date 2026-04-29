from pathlib import Path

import pandas as pd

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


def extract_synth_vuln_fixes(data_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from synth-vuln-fixes.

    Source: HuggingFace `patched-codes/synth-vuln-fixes`.

    Expected Parquet schema (columns probed in priority order):
      - cwe / cwe_id / weakness      : CWE identifier
      - vulnerable_code / vuln_code  : vulnerable function
      - fixed_code / patched_code    : fixed function
      - language                     : programming language (filter: Python)

    Positives: vulnerable_code with non-empty CWE.
    Negatives: fixed_code, cwes=[].
    """
    if data_path.is_dir():
        parts = sorted(data_path.rglob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No .parquet files found in {data_path}")
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    else:
        df = pd.read_parquet(data_path)

    cols = set(df.columns)

    # CWE column
    cwe_col = next((c for c in ("cwe", "cwe_id", "weakness", "CWE") if c in cols), None)
    # Code columns
    vuln_col  = next((c for c in ("vulnerable_code", "vuln_code", "vulnerable") if c in cols), None)
    fixed_col = next((c for c in ("fixed_code", "patched_code", "fixed") if c in cols), None)
    # Language column
    lang_col  = next((c for c in ("language", "lang", "Language") if c in cols), None)

    if not vuln_col:
        raise ValueError(f"Could not find vulnerable code column in {data_path}. Columns: {list(cols)}")

    # Language filter: keep Python rows if column present
    if lang_col:
        df = df[df[lang_col].str.lower().str.strip() == "python"].copy()

    samples: list[FunctionSample] = []

    for _, row in df.iterrows():
        cwe_raw = str(row[cwe_col]) if cwe_col else ""
        cwe = _normalise_cwe(cwe_raw)
        if not cwe or cwe in _NVD_PLACEHOLDERS:
            continue

        vuln = str(row[vuln_col] or "").strip()
        if vuln:
            samples.append(FunctionSample(
                code=vuln, cwes=[cwe], label=1,
                branch="ai", language="Python",
            ))

        if fixed_col:
            fixed = str(row[fixed_col] or "").strip()
            if fixed:
                samples.append(FunctionSample(
                    code=fixed, cwes=[], label=0,
                    branch="ai", language="Python",
                ))

    return samples

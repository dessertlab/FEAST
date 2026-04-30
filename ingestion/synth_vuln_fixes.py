import re
from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample
from ingestion.utils import _CWE_RE, _NVD_PLACEHOLDERS

_CODE_BLOCK_RE = re.compile(r"Original Code:.*?```[^\n]*\n(.*?)```", re.DOTALL)


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

    Each row has a single `messages` column with 3 entries:
      [0] system prompt
      [1] user: vulnerability report (contains CWE) + original vulnerable code block
      [2] assistant: fixed code

    Positives: code from "Original Code:" block, CWE from report.
    Negatives: assistant fixed code, cwes=[].
    """
    if data_path.is_dir():
        parts = sorted(data_path.rglob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No .parquet files found in {data_path}")
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    else:
        df = pd.read_parquet(data_path)

    samples: list[FunctionSample] = []

    for _, row in df.iterrows():
        msgs = row["messages"]
        if len(msgs) < 3:
            continue
        user_text = msgs[1].get("content", "")
        fixed_code = str(msgs[2].get("content", "") or "").strip()

        cwe = _normalise_cwe(user_text)
        if not cwe or cwe in _NVD_PLACEHOLDERS:
            continue

        m = _CODE_BLOCK_RE.search(user_text)
        vuln_code = m.group(1).strip() if m else ""

        if vuln_code:
            samples.append(FunctionSample(
                code=vuln_code, cwes=[cwe], label=1,
                branch="ai", language="Python",
            ))
        if fixed_code:
            samples.append(FunctionSample(
                code=fixed_code, cwes=[], label=0,
                branch="ai", language="Python",
            ))

    return samples

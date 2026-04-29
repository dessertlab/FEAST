from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample
from ingestion.utils import _NVD_PLACEHOLDERS, _CWE_RE

# CrossVul language column values -> FunctionSample language labels
_LANG_MAP = {
    "c":      "C/C++",
    "c++":    "C/C++",
    "cpp":    "C/C++",
    "java":   "Java",
    "python": "Python",
}


def extract_crossvul(
    data_path: Path,
    language: str = "C/C++",
) -> list[FunctionSample]:
    """Extract FunctionSamples from CrossVul Parquet file(s).

    Source: HuggingFace `hitoshura25/crossvul`.
    Columns used: language, cwe_id, fixed_code.

    Returns only label=0 (fixed/safe) samples; CrossVul does not provide
    unambiguous per-function vulnerable labels (the vulnerable_code column
    exists but CWE is at commit level and multi-function commits are common).

    language parameter accepts: "C/C++", "Java", "Python".
    """
    if data_path.is_dir():
        parts = sorted(data_path.glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No .parquet files found in {data_path}")
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    else:
        df = pd.read_parquet(data_path)

    # Normalise language column to match our labels
    df["_lang"] = df["language"].str.lower().str.strip().map(_LANG_MAP)
    df = df[df["_lang"] == language].copy()

    # CWE filter
    df = df[df["cwe_id"].notna()].copy()
    df = df[df["cwe_id"].str.strip() != ""].copy()
    df = df[~df["cwe_id"].str.strip().isin(_NVD_PLACEHOLDERS)].copy()
    df["cwes_parsed"] = df["cwe_id"].str.strip().apply(_CWE_RE.findall)
    df = df[df["cwes_parsed"].map(len) > 0].copy()

    # Code filter
    df = df[df["fixed_code"].notna() & (df["fixed_code"].str.strip() != "")].copy()

    return [
        FunctionSample(
            code=str(row["fixed_code"]),
            cwes=[],
            label=0,
            branch="real",
            language=language,
        )
        for _, row in df.iterrows()
    ]

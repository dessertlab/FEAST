from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample
from ingestion.utils import _NVD_PLACEHOLDERS


def _normalize_cwe(raw: str) -> str:
    """'cwe-022' -> 'CWE-22', strips leading zeros and normalises case."""
    raw = raw.strip()
    if raw.lower().startswith("cwe-"):
        num_str = raw[4:]
        try:
            return f"CWE-{int(num_str)}"
        except ValueError:
            return ""
    return ""


def _lang_from_filename(filename: str) -> str | None:
    """Detect language from file extension."""
    suffix = Path(filename).suffix.lower()
    if suffix in (".c", ".cpp", ".cc", ".h", ".hpp"):
        return "C/C++"
    if suffix == ".py":
        return "Python"
    return None


def extract_sven(
    data_path: Path,
    language: str = "C/C++",
) -> list[FunctionSample]:
    """Extract FunctionSamples from SVEN Parquet file(s).

    Source: HuggingFace `bstee615/sven`.
    Columns used: file_name, func_src_before, func_src_after, vul_type.

    label=1: func_src_before (vulnerable version), CWE from vul_type.
    label=0: func_src_after  (fixed/safe version),  cwes=[].

    language parameter accepts: "C/C++", "Python".
    """
    if data_path.is_dir():
        parts = sorted(data_path.rglob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No .parquet files found in {data_path}")
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    else:
        df = pd.read_parquet(data_path)

    # Language filter from file_name extension
    df["_lang"] = df["file_name"].apply(_lang_from_filename)
    df = df[df["_lang"] == language].copy()

    samples: list[FunctionSample] = []

    for _, row in df.iterrows():
        cwe_raw = str(row.get("vul_type", "") or "")
        cwe = _normalize_cwe(cwe_raw)
        if not cwe or cwe in _NVD_PLACEHOLDERS:
            continue

        # Positive: vulnerable function before the fix
        before = str(row.get("func_src_before", "") or "").strip()
        if before:
            samples.append(FunctionSample(
                code=before, cwes=[cwe], label=1,
                branch="real", language=language,
            ))

        # Negative: fixed function after the patch
        after = str(row.get("func_src_after", "") or "").strip()
        if after:
            samples.append(FunctionSample(
                code=after, cwes=[], label=0,
                branch="real", language=language,
            ))

    return samples

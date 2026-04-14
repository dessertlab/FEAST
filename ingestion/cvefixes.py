from collections import Counter
from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample

_C_LANGUAGES = {"C", "C++"}
_NVD_PLACEHOLDERS = {"NVD-CWE-Other", "NVD-CWE-noinfo", "NVD-CWE-Other "}


def extract_cvefixes(data_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from CVEfixes Parquet file(s).

    Source: Hugging Face `hitoshura25/cvefixes` (3 Parquet shards).
    Pass either a directory containing the `.parquet` files or a single file.

    Columns used: vulnerable_code, cwe_id, hash, language.

    Positives: rows where
      - language ∈ {"C", "C++"}
      - vulnerable_code is non-empty
      - cwe_id is non-empty and not an NVD placeholder
      - commit hash appears in exactly one row (single-function commit filter)

    Negatives: none — CVEfixes does not provide explicitly labeled safe samples.
    """
    if data_path.is_dir():
        parts = sorted(data_path.glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No .parquet files found in {data_path}")
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    else:
        df = pd.read_parquet(data_path)

    # Language filter
    df = df[df["language"].isin(_C_LANGUAGES)].copy()

    # Code filter
    df = df[df["vulnerable_code"].notna() & (df["vulnerable_code"].str.strip() != "")].copy()

    # CWE filter
    df = df[df["cwe_id"].notna()].copy()
    df = df[df["cwe_id"].str.strip() != ""].copy()
    df = df[~df["cwe_id"].str.strip().isin(_NVD_PLACEHOLDERS)].copy()

    # Single-function commit filter
    hash_counts = Counter(df["hash"])
    single_hashes = {h for h, n in hash_counts.items() if n == 1}
    df = df[df["hash"].isin(single_hashes)]

    return [
        FunctionSample(
            code=str(row["vulnerable_code"]),
            cwes=[str(row["cwe_id"]).strip()],
            label=1,
        )
        for _, row in df.iterrows()
    ]

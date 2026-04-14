from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample


@dataclass
class RawICVulEntry:
    code: str
    cwes: list[str]
    label: int    # always 1 for ICVul (VCC-identified)
    is_vcc: bool  # always True after inner join; kept for schema documentation


def extract_icvul(base_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from an ICVul dataset directory.

    Expects two CSV files in base_path:
      - functions.csv         columns: func_id, func_before
      - cve_fc_vcc_mapping.csv  columns: cwe_id, vcc_func_id

    If the actual column names differ from the above, update the rename
    calls below — the join and filter logic does not change.

    Positives: functions present in cve_fc_vcc_mapping (VCC-identified).
    Negatives: none — ICVul does not provide explicitly labeled safe samples.
    """
    functions = pd.read_csv(base_path / "functions.csv")
    mapping = pd.read_csv(base_path / "cve_fc_vcc_mapping.csv")

    # Normalize column names to internal names
    functions = functions.rename(columns={"func_before": "code"})
    mapping = mapping.rename(columns={"vcc_func_id": "func_id"})

    # Group CWEs per function (a function can appear in multiple CVEs)
    cwe_by_func = (
        mapping.groupby("func_id")["cwe_id"]
        .apply(lambda s: sorted(set(s.dropna().astype(str))))
        .reset_index()
        .rename(columns={"cwe_id": "cwes"})
    )

    # Join: keep only VCC functions (inner join drops non-VCC)
    merged = functions.merge(cwe_by_func, on="func_id", how="inner")

    # Drop entries with no CWE resolved
    merged = merged[merged["cwes"].map(len) > 0]

    return [
        FunctionSample(code=str(row["code"]), cwes=list(row["cwes"]), label=1)
        for _, row in merged.iterrows()
    ]

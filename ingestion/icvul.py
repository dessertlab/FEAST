import ast
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample

# CWE values that are not real CWE IDs — used by NVD when the type is unclear
_NVD_PLACEHOLDERS = {"NVD-CWE-Other", "NVD-CWE-noinfo", "NVD-CWE-Other "}


@dataclass
class RawICVulEntry:
    code: str
    cwes: list[str]
    label: int    # always 1 for ICVul (VCC-identified)
    commit_hash: str


def _parse_vcc_hashes(raw) -> list[str]:
    """Parse vcc_hash field, which is stored as a JSON/Python list string."""
    if raw is None or (isinstance(raw, float)):
        return []
    raw = str(raw).strip()
    if not raw or raw == "[]":
        return []
    try:
        items = ast.literal_eval(raw)
        return [str(h).strip() for h in items if h]
    except (ValueError, SyntaxError):
        return []


def extract_icvul(base_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from an ICVul dataset directory.

    Actual file layout (from the released dataset):
      - function_info.csv        columns: hash, code, before_change, ...
      - cve_fc_vcc_mapping.csv   columns: cve_id, cwe_id, vcc_hash, ...

    Join logic:
      1. Parse vcc_hash (JSON list of commit hashes) from the mapping table.
      2. Match commit hashes to function_info rows where before_change=True
         (the pre-fix, i.e. vulnerable, version of the function).
      3. Assign the CVE's CWE to each matched function.
      4. Drop entries whose CWE resolves to an NVD placeholder (not a real CWE).

    Positives: VCC-identified, before_change=True functions with a real CWE.
    Negatives: none — ICVul does not provide explicitly labeled safe samples.
    """
    functions = pd.read_csv(base_path / "function_info.csv")
    mapping   = pd.read_csv(base_path / "cve_fc_vcc_mapping.csv")

    # Keep only the vulnerable version of each function
    functions = functions[functions["before_change"] == True].copy()

    # Explode vcc_hash lists → one row per (cwe_id, vcc_hash)
    records = []
    for _, row in mapping.iterrows():
        cwe_id = str(row.get("cwe_id", "")).strip()
        if not cwe_id or cwe_id in _NVD_PLACEHOLDERS:
            continue
        for h in _parse_vcc_hashes(row.get("vcc_hash")):
            records.append({"hash": h, "cwe_id": cwe_id})

    if not records:
        return []

    vcc_df = pd.DataFrame(records)

    # Group CWEs per VCC commit hash
    cwe_by_hash = (
        vcc_df.groupby("hash")["cwe_id"]
        .apply(lambda s: sorted(set(s.dropna().astype(str))))
        .reset_index()
        .rename(columns={"cwe_id": "cwes"})
    )

    # Join: inner join keeps only functions at VCC commits
    merged = functions.merge(cwe_by_hash, on="hash", how="inner")
    merged = merged[merged["cwes"].map(len) > 0]

    return [
        FunctionSample(code=str(row["code"]), cwes=list(row["cwes"]), label=1)
        for _, row in merged.iterrows()
    ]

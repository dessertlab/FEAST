from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample

# CWE values that are not real CWE IDs -- used by NVD when the type is unclear
_NVD_PLACEHOLDERS = {"NVD-CWE-Other", "NVD-CWE-noinfo", "NVD-CWE-Other "}


@dataclass
class RawICVulEntry:
    code: str
    cwes: list[str]
    label: int    # always 1 for ICVul (VCC-identified)
    commit_hash: str


def extract_icvul(base_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from an ICVul dataset directory.

    Actual file layout (from the released dataset):
      - function_info.csv        columns: hash (= fc_hash), code, before_change, ...
      - cve_fc_vcc_mapping.csv   columns: cve_id, cwe_id, fc_hash, vcc_hash, ...

    Join logic:
      1. Filter function_info to rows where before_change=True
         (the pre-fix, i.e. vulnerable, version of the function).
      2. Join function_info.hash -> cve_fc_vcc_mapping.fc_hash.
         (fc_hash is the hash of the function change, NOT the VCC commit hash.)
      3. Assign the CVE's CWE to each matched function.
      4. Drop entries whose CWE resolves to an NVD placeholder.

    Positives: before_change=True functions with a real CWE.
    Negatives: none -- ICVul does not provide explicitly labeled safe samples.
    """
    functions = pd.read_csv(base_path / "function_info.csv")
    mapping   = pd.read_csv(base_path / "cve_fc_vcc_mapping.csv")

    # Keep only the vulnerable version of each function
    functions = functions[functions["before_change"] == True].copy()

    # Filter out NVD placeholder CWEs from mapping
    mapping = mapping[mapping["cwe_id"].notna()].copy()
    mapping = mapping[~mapping["cwe_id"].str.strip().isin(_NVD_PLACEHOLDERS)].copy()
    mapping = mapping[mapping["cwe_id"].str.strip() != ""].copy()

    # Group CWEs per fc_hash
    cwe_by_hash = (
        mapping.groupby("fc_hash")["cwe_id"]
        .apply(lambda s: sorted(set(s.dropna().astype(str).str.strip())))
        .reset_index()
        .rename(columns={"cwe_id": "cwes", "fc_hash": "hash"})
    )

    # Join: function_info.hash -> fc_hash
    merged = functions.merge(cwe_by_hash, on="hash", how="inner")
    merged = merged[merged["cwes"].map(len) > 0]

    return [
        FunctionSample(code=str(row["code"]), cwes=list(row["cwes"]), label=1)
        for _, row in merged.iterrows()
    ]

import ast
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample


@dataclass
class RawPrimeVulEntry:
    code: str
    cwes: list[str]
    label: int
    commit_id: str
    project: str


def _parse_cwes(raw) -> list[str]:
    """Parse CWE field from PrimeVul row.

    PrimeVul's 'cwe' column can be:
      - NaN / None / empty string  → []
      - "CWE-119"                  → ["CWE-119"]
      - "['CWE-119', 'CWE-787']"   → ["CWE-119", "CWE-787"]
    """
    if raw is None or isinstance(raw, float):  # catches NaN
        return []
    raw = str(raw).strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            items = ast.literal_eval(raw)
            return [c.strip() for c in items if isinstance(c, str) and c.strip()]
        except (ValueError, SyntaxError):
            return []
    return [c.strip() for c in raw.split() if c.strip()]


def _load_raw(path: Path) -> list[RawPrimeVulEntry]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_json(path)

    entries = []
    for _, row in df.iterrows():
        entries.append(RawPrimeVulEntry(
            code=str(row["func"]),
            cwes=_parse_cwes(row.get("cwe")),
            label=int(row["target"]),
            commit_id=str(row.get("commit_id", "")),
            project=str(row.get("project", "")),
        ))
    return entries


def _single_function_commit_ids(entries: list[RawPrimeVulEntry]) -> set[str]:
    """Return commit IDs where exactly one target=1 function was changed."""
    counts = Counter(e.commit_id for e in entries if e.label == 1)
    return {cid for cid, n in counts.items() if n == 1}


def extract_primevul(path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from a PrimeVul JSON or Parquet file.

    Positives: target=1, CWE non-empty, single-function commit.
    Negatives: all target=0 entries (explicitly safe by PrimeVul's construction).
    """
    entries = _load_raw(path)
    single_commits = _single_function_commit_ids(entries)

    samples = []
    for e in entries:
        if e.label == 1:
            if e.cwes and e.commit_id in single_commits:
                samples.append(FunctionSample(code=e.code, cwes=e.cwes, label=1))
        else:
            samples.append(FunctionSample(code=e.code, cwes=[], label=0))
    return samples

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample


@dataclass
class RawMegaVulEntry:
    code: str
    cwes: list[str]
    label: int
    commit_id: str
    cvss_score: float | None


def _parse_cwe(raw) -> list[str]:
    if raw is None or isinstance(raw, float):
        return []
    val = str(raw).strip()
    return [val] if val else []


def extract_megavul(path: Path, cvss_threshold: float | None = None) -> list[FunctionSample]:
    """Extract FunctionSamples from a MegaVul JSON file.

    Positives: target=1, CWE non-empty, single-function commit.
    If cvss_threshold is set, entries with CVSS Score < threshold or null CVSS
    are also filtered out.

    Negatives: MegaVul does not provide explicitly labeled safe samples.
    func_after is not used as a negative. This collection contributes positives only.
    """
    df = pd.read_json(path)

    entries = [
        RawMegaVulEntry(
            code=str(row["func_before"]),
            cwes=_parse_cwe(row.get("CWE ID")),
            label=int(row.get("target", 1)),
            commit_id=str(row.get("commit_id", "")),
            cvss_score=float(row["CVSS Score"]) if pd.notna(row.get("CVSS Score")) else None,
        )
        for _, row in df.iterrows()
    ]

    counts = Counter(e.commit_id for e in entries if e.label == 1)
    single_commits = {cid for cid, n in counts.items() if n == 1}

    samples = []
    for e in entries:
        if e.label != 1:
            continue
        if not e.cwes:
            continue
        if e.commit_id not in single_commits:
            continue
        if cvss_threshold is not None:
            if e.cvss_score is None or e.cvss_score < cvss_threshold:
                continue
        samples.append(FunctionSample(code=e.code, cwes=e.cwes, label=1))
    return samples

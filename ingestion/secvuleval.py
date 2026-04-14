import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample


@dataclass
class RawSecVulEvalEntry:
    code: str
    label: int
    cve_id: str
    cwes: list[str]  # populated after NVD cache lookup; empty if unresolvable


def extract_secvuleval(data_path: Path, nvd_cache_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from a SecVulEval CSV/JSON file.

    Positives: label=1 entries where the NVD cache resolves cve_id to at least
    one CWE. Entries with no resolvable CWE are dropped.

    Negatives: all label=0 entries. CWE lookup is not applied to negatives;
    they carry cwes=[].

    Args:
        data_path:      Path to the SecVulEval CSV or JSON file.
                        Expected columns: func, label, cve_id.
        nvd_cache_path: Path to a JSON file mapping CVE IDs to lists of CWE strings.
                        Format: {"CVE-YYYY-NNNN": ["CWE-XXX", ...], ...}
    """
    nvd_cache: dict[str, list[str]] = json.loads(nvd_cache_path.read_text())

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        df = pd.read_json(data_path)

    entries = [
        RawSecVulEvalEntry(
            code=str(row["func"]),
            label=int(row["label"]),
            cve_id=str(row["cve_id"]),
            cwes=nvd_cache.get(str(row["cve_id"]), []),
        )
        for _, row in df.iterrows()
    ]

    samples = []
    for e in entries:
        if e.label == 1:
            if e.cwes:
                samples.append(FunctionSample(code=e.code, cwes=e.cwes, label=1))
        else:
            samples.append(FunctionSample(code=e.code, cwes=[], label=0))
    return samples

from pathlib import Path

import pandas as pd

from ingestion.schema import FunctionSample

# Mapping from ESBMC/FormAI vul_type strings to CWE IDs.
# Based on ESBMC property checks used in the FormAI dataset paper.
_VUL_TYPE_TO_CWE: dict[str, str] = {
    "buffer overflow":              "CWE-121",
    "stack-based buffer overflow":  "CWE-121",
    "heap buffer overflow":         "CWE-122",
    "out-of-bounds":                "CWE-787",
    "out-of-bounds read":           "CWE-125",
    "out-of-bounds write":          "CWE-787",
    "memory leak":                  "CWE-401",
    "use after free":               "CWE-416",
    "use-after-free":               "CWE-416",
    "double free":                  "CWE-415",
    "dangling pointer":             "CWE-416",
    "null pointer dereference":     "CWE-476",
    "null pointer":                 "CWE-476",
    "integer overflow":             "CWE-190",
    "integer underflow":            "CWE-191",
    "division by zero":             "CWE-369",
    "array bounds":                 "CWE-119",
    "array out of bounds":          "CWE-119",
    "uninitialized variable":       "CWE-457",
    "format string":                "CWE-134",
    "race condition":               "CWE-362",
    "deadlock":                     "CWE-833",
    "arithmetic overflow":          "CWE-190",
    "signed integer overflow":      "CWE-190",
    "unsigned integer overflow":    "CWE-190",
}


def _map_vul_type(raw: str) -> str | None:
    if not raw:
        return None
    key = raw.strip().lower()
    return _VUL_TYPE_TO_CWE.get(key)


def extract_formai(data_path: Path) -> list[FunctionSample]:
    """Extract FunctionSamples from FormAI CSV file.

    Source: GitHub / HuggingFace `NTUYG/FormAI-dataset`.
    Expected file: a CSV with columns: filename, classification, source_code, vul_type.

    Positives: classification contains 'Vulnerable', vul_type maps to a known CWE.
    Negatives: classification contains 'Non-Vulnerable' or 'Safe'.

    Only C/C++ files (extension filter on filename column).
    """
    if data_path.is_dir():
        candidates = list(data_path.glob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No .csv files found in {data_path}")
        data_path = candidates[0]

    df = pd.read_csv(data_path)

    samples: list[FunctionSample] = []
    for _, row in df.iterrows():
        filename = str(row.get("filename", "") or "")
        ext = Path(filename).suffix.lower()
        if ext not in (".c", ".cpp", ".cc"):
            continue

        code = str(row.get("source_code", "") or "").strip()
        if not code:
            continue

        classification = str(row.get("classification", "") or "").strip().lower()
        vul_type = str(row.get("vul_type", "") or "").strip()

        if "vulnerable" in classification and "non" not in classification:
            cwe = _map_vul_type(vul_type)
            if not cwe:
                continue
            samples.append(FunctionSample(
                code=code, cwes=[cwe], label=1,
                branch="ai", language="C/C++",
            ))
        elif "non-vulnerable" in classification or "safe" in classification:
            samples.append(FunctionSample(
                code=code, cwes=[], label=0,
                branch="ai", language="C/C++",
            ))

    return samples

import re

_NVD_PLACEHOLDERS: frozenset[str] = frozenset({
    "NVD-CWE-Other",
    "NVD-CWE-noinfo",
    "NVD-CWE-Other ",
})

_CWE_RE: re.Pattern = re.compile(r"CWE-\d+")


def split_cwe(raw: str) -> list[str]:
    """Extract all CWE-NNN tokens from raw string.

    Handles concatenated IDs (CWE-20CWE-190 -> [CWE-20, CWE-190]),
    NVD placeholders (dropped), and empty/None input.
    """
    if not raw or not isinstance(raw, str):
        return []
    raw = raw.strip()
    if raw in _NVD_PLACEHOLDERS:
        return []
    return _CWE_RE.findall(raw)

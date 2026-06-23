import hashlib
import re

_NVD_PLACEHOLDERS: frozenset[str] = frozenset({
    "NVD-CWE-Other",
    "NVD-CWE-noinfo",
    "NVD-CWE-Other ",
})

_CWE_RE: re.Pattern = re.compile(r"CWE-\d+")


def _norm_code(code: str) -> str:
    """Normalise code for content hashing (mirrors ``main._norm``)."""
    code = str(code).replace("\r\n", "\n").replace("\r", "\n")
    lines = [l.rstrip() for l in code.split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def code_sample_id(code: str) -> str:
    """Stable content-derived sample id (matches ``main._hash(code)[:16]``)."""
    return hashlib.sha256(_norm_code(code).encode()).hexdigest()[:16]


# Samples deliberately dropped at ingestion time. Keyed by content-derived
# sample_id so the exclusion survives any path/source bookkeeping.
#   6792fa11e2e1c154 -- CVEfixes(C) RELIC crypto benchmark function (CWE-190)
#       that hangs the SAT static-analysis tools (joern/codeql run unbounded and
#       never terminate), so it can never be materialized/analysed.
EXCLUDED_SAMPLE_IDS: frozenset[str] = frozenset({
    "6792fa11e2e1c154",
})


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

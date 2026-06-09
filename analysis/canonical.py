"""Canonicalisation of CWE IDs to MITRE CWE-1000 family levels.

The fusion methodology evaluates tools and ground truth at the granularity of a
*CWE family* rather than the raw leaf CWE. A family is the ancestor of a CWE at a
chosen abstraction level of the Research Concepts view (CWE-1000), reached by
climbing the deterministic **primary** ChildOf path (``View_ID=1000``,
``Ordinal=Primary``). Three levels are supported:

* ``pillar``       — the top weakness of the view (≈10 buckets, very coarse).
* ``subcategory``  — the node directly under the pillar (coarse).
* ``class``        — the nearest ancestor (incl. self) tagged ``Abstraction=Class``;
                     finest of the three. This is the recommended default: it keeps
                     SQLi/XSS/command-injection distinct while merging e.g. buffer
                     over-read/over-write into CWE-119.

Once tool outputs and ground truth are mapped to families, downstream matching is a
plain *exact* set membership — the permissive vertical matching is gone; the only
place the CWE hierarchy is consulted is here.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from analysis.dataset import as_list, normalise_cwe, normalise_cwes
from ingestion.cwe_navigator import CWENavigator

LEVELS = ("pillar", "subcategory", "class")
RESEARCH_VIEW = "1000"


class CweCanonicalizer:
    """Map raw CWE IDs to a family at one CWE-1000 abstraction level."""

    def __init__(self, navigator: CWENavigator, level: str = "class", view: str = RESEARCH_VIEW):
        if level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
        self.nav = navigator
        self.level = level
        self.view = view
        self._cache: dict[str, str | None] = {}

    @classmethod
    def from_xml(cls, level: str = "class", cwe_xml_path: str = "data/cwec_latest.xml") -> "CweCanonicalizer":
        return cls(CWENavigator(cwe_xml_path), level=level)

    # ── core mapping ──────────────────────────────────────────────────────────
    def family(self, cwe) -> str | None:
        """Return the family CWE-ID for ``cwe`` at this canonicaliser's level.

        Returns ``None`` for tokens that are not weaknesses in the catalogue
        (categories, deprecated-only ids, junk like ``CWE-0``) — these are reported
        as *unmapped* and excluded from the analysis.
        """
        normalised = normalise_cwe(cwe)
        if normalised is None:
            return None
        num = normalised.removeprefix("CWE-")
        if num in self._cache:
            return self._cache[num]

        family = self._resolve(num)
        self._cache[num] = family
        return family

    def _resolve(self, num: str) -> str | None:
        # Only weaknesses live in the CWE-1000 ChildOf hierarchy.
        if num not in self.nav.weaknesses:
            return None
        # primary_path is leaf-first: (cwe, parent, ..., pillar).
        path = self.nav.primary_path(num, self.view)
        pillar = path[-1]

        if self.level == "pillar":
            target = pillar
        elif self.level == "subcategory":
            # Node directly under the pillar; if the CWE *is* the pillar, it maps to itself.
            target = path[-2] if len(path) >= 2 else pillar
        else:  # "class": nearest Class ancestor (incl. self), else fall back to the pillar.
            target = next(
                (node for node in path if self.nav.abstraction(node) == "Class"),
                pillar,
            )
        return f"CWE-{target}"

    def families(self, cwes: Iterable) -> list[str]:
        """Map an iterable of raw CWEs to the sorted set of their (defined) families."""
        out = {self.family(cwe) for cwe in normalise_cwes(cwes)}
        out.discard(None)
        return sorted(out, key=lambda c: int(c.removeprefix("CWE-")))

    # ── dataframe-level transform ─────────────────────────────────────────────
    def canonicalize_frame(self, df: pd.DataFrame, tool_columns: Iterable[str]) -> pd.DataFrame:
        """Return a copy of ``df`` with GT ``cwes`` and every tool column rewritten
        from lists of raw CWE-IDs to sorted lists of family CWE-IDs.

        Computed once up front and reused across folds, so the (cached) hierarchy
        lookups happen a single time per distinct CWE.
        """
        out = df.copy()
        columns = ["cwes", *[c for c in tool_columns if c in out.columns]]
        for column in columns:
            if column in out.columns:
                out[column] = out[column].map(lambda values: self.families(as_list(values)))
        return out

    def canonical_map(self, cwes: Iterable) -> pd.DataFrame:
        """Audit table mapping each distinct input CWE to its family (and level)."""
        rows = []
        for cwe in sorted(normalise_cwes(cwes), key=lambda c: int(c.removeprefix("CWE-"))):
            rows.append({"cwe": cwe, "family": self.family(cwe), "level": self.level})
        return pd.DataFrame(rows, columns=["cwe", "family", "level"])


def all_raw_cwes(df: pd.DataFrame, tool_columns: Iterable[str]) -> set[str]:
    """Collect every raw CWE appearing in ground truth or any tool column."""
    cwes: set[str] = set()
    for column in ["cwes", *[c for c in tool_columns if c in df.columns]]:
        if column in df.columns:
            for values in df[column]:
                cwes |= normalise_cwes(as_list(values))
    return cwes

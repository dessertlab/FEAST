"""Traditional K-of-N voting baseline.

The unweighted reference the paper argues against: a family is predicted vulnerable when
at least ``threshold`` of the considered tools fire it. With ``supported_only`` the
denominator is restricted to tools that ever fired the family in calibration (so a tool
that cannot detect the family does not dilute the vote).
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from analysis.fusion.common import evidence_row, is_supported, sample_ids_of


def traditional_vote_predictions(
    df: pd.DataFrame,
    tools: Sequence[str],
    families: Sequence[str],
    threshold: int,
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
    lookup: dict[tuple[str, str], dict] | None = None,
    supported_only: bool = False,
) -> pd.DataFrame:
    fire_sets, fired_union = fire_index
    sample_ids = sample_ids_of(df)
    strategy = (
        f"traditional_{threshold}_of_supported" if supported_only
        else f"traditional_{threshold}_of_{len(tools)}"
    )

    # Considered tools depend only on the family, not the row.
    considered: dict[str, list[str]] = {}
    for family in families:
        if supported_only:
            considered[family] = [t for t in tools if is_supported((lookup or {}).get((t, family)))]
        else:
            considered[family] = list(tools)

    rows: list[dict] = []
    for row_index in range(len(df)):
        fired_here = fired_union[row_index]
        for family in families:
            cons = considered[family]
            if family not in fired_here:
                fire_count = 0
            else:
                fire_count = sum(1 for tool in cons if family in fire_sets[(row_index, tool)])
            n = len(cons)
            rows.append(evidence_row(
                sample_ids[row_index], row_index, family, strategy,
                prediction=(fire_count >= threshold),
                score=(0.0 if n == 0 else fire_count / n),
                vuln=float(fire_count), safe=float(n - fire_count),
                k_supported=n, k_fired=fire_count, k_abstained=len(tools) - n,
                label=labels[(row_index, family)],
            ))
    return pd.DataFrame(rows)

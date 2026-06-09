"""Noisy-OR fusion.

A structured probabilistic combiner that matches the OR-recall intuition: each tool that
fires a family independently "explains" the vulnerability with activation probability
``q_t`` (its calibration PPV — P(vulnerable | this tool fired)), plus a background leak
``π`` (the family base rate, i.e. P(vulnerable | no tool fired)):

    P(vulnerable) = 1 - (1 - π) · ∏_{t fired} (1 - q_t)

Uses only the calibration reliabilities (no extra fitting), like the DST / naive-Bayes
strategies. More firing tools push the score up multiplicatively, never down.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from analysis.fusion.common import evidence_row, is_supported, metric_value, sample_ids_of


def noisy_or_predictions(
    df: pd.DataFrame,
    lookup: dict[tuple[str, str], dict],
    tools: Sequence[str],
    families: Sequence[str],
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
    tau: float = 0.5,
    eps: float = 1e-3,
) -> pd.DataFrame:
    fire_sets, _ = fire_index
    sample_ids = sample_ids_of(df)
    n = len(df)

    rows: list[dict] = []
    for family in families:
        sup, q = [], []
        prior = 0.0
        for tool in tools:
            row = lookup.get((tool, family))
            if not is_supported(row):
                continue
            ppv = metric_value(row, "ppv")
            sup.append(tool)
            q.append(min(max(0.0 if ppv is None else ppv, eps), 1.0 - eps))
            total = float(row["tp"]) + float(row["fp"]) + float(row["tn"]) + float(row["fn"])
            prior = (float(row["tp"]) + float(row["fn"])) / total if total else 0.0
        leak = min(max(prior, eps), 1.0 - eps)
        abstained = len(tools) - len(sup)

        q_arr = np.asarray(q)
        for i in range(n):
            fired_mask = np.array([family in fire_sets[(i, t)] for t in sup], dtype=bool)
            score = 1.0 - (1.0 - leak) * np.prod(1.0 - q_arr[fired_mask]) if len(sup) else leak
            rows.append(evidence_row(
                sample_ids[i], i, family, "noisy_or",
                prediction=bool(score >= tau), score=float(score),
                vuln=float(score), safe=float(1.0 - score),
                k_supported=len(sup), k_fired=int(fired_mask.sum()), k_abstained=abstained,
                label=labels[(i, family)],
            ))
    return pd.DataFrame(rows)

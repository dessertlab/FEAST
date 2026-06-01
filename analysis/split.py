
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Iterable

import pandas as pd

from analysis.dataset import as_list, normalise_cwes

SAFE_LABEL = "SAFE"


def row_multilabels(value, safe_label: str = SAFE_LABEL) -> set[str]:
    labels = normalise_cwes(as_list(value))
    return labels or {safe_label}


def _label_matrix(labels_by_row: list[set[str]]) -> tuple[list[str], list[list[int]]]:
    labels = sorted({label for row_labels in labels_by_row for label in row_labels})
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0] * len(labels) for _ in labels_by_row]
    for row_id, row_labels in enumerate(labels_by_row):
        for label in row_labels:
            matrix[row_id][index[label]] = 1
    return labels, matrix


def _iterstrat_folds(labels_by_row: list[set[str]], n_splits: int, random_state: int) -> list[int] | None:
    try:
        import numpy as np
        from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    except ImportError:
        return None

    _labels, matrix = _label_matrix(labels_by_row)
    x = np.zeros((len(labels_by_row), 1))
    y = np.array(matrix)
    splitter = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = [0] * len(labels_by_row)
    for fold, (_train_idx, test_idx) in enumerate(splitter.split(x, y)):
        for row_id in test_idx:
            folds[int(row_id)] = fold
    return folds


def _fallback_iterative_folds(labels_by_row: list[set[str]], n_splits: int, random_state: int) -> list[int]:
    rng = random.Random(random_state)
    unassigned = set(range(len(labels_by_row)))
    folds = [-1] * len(labels_by_row)
    fold_sizes = [0] * n_splits
    fold_label_counts = [Counter() for _ in range(n_splits)]

    rows_by_label: dict[str, list[int]] = defaultdict(list)
    for row_id, labels in enumerate(labels_by_row):
        for label in labels:
            rows_by_label[label].append(row_id)

    labels_by_rarity = sorted(rows_by_label, key=lambda label: (len(rows_by_label[label]), label))

    def choose_fold(row_id: int) -> int:
        labels = labels_by_row[row_id]

        def score(fold: int) -> tuple[int, int, float]:
            label_load = sum(fold_label_counts[fold][label] for label in labels)
            return (label_load, fold_sizes[fold], rng.random())

        return min(range(n_splits), key=score)

    for label in labels_by_rarity:
        candidates = [row_id for row_id in rows_by_label[label] if row_id in unassigned]
        rng.shuffle(candidates)
        for row_id in candidates:
            fold = choose_fold(row_id)
            folds[row_id] = fold
            fold_sizes[fold] += 1
            fold_label_counts[fold].update(labels_by_row[row_id])
            unassigned.remove(row_id)

    leftovers = list(unassigned)
    rng.shuffle(leftovers)
    for row_id in leftovers:
        fold = min(range(n_splits), key=lambda f: (fold_sizes[f], rng.random()))
        folds[row_id] = fold
        fold_sizes[fold] += 1
        fold_label_counts[fold].update(labels_by_row[row_id])

    return folds


def multilabel_stratified_kfold(
    df: pd.DataFrame,
    n_splits: int = 5,
    label_column: str = "cwes",
    id_column: str = "sample_id",
    safe_label: str = SAFE_LABEL,
    random_state: int = 0,
) -> pd.DataFrame:
    """Assign rows to multilabel-stratified folds using CWE labels.

    If the optional ``iterative-stratification`` package is installed, this uses
    its MultilabelStratifiedKFold implementation. Otherwise it falls back to a
    deterministic rarity-first iterative assignment that keeps rare CWEs spread
    across folds as evenly as possible.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if len(df) < n_splits:
        raise ValueError("n_splits cannot exceed the number of rows")
    if label_column not in df.columns:
        raise ValueError(f"missing label column: {label_column}")

    labels_by_row = [row_multilabels(value, safe_label=safe_label) for value in df[label_column]]
    folds = _iterstrat_folds(labels_by_row, n_splits, random_state)
    if folds is None:
        folds = _fallback_iterative_folds(labels_by_row, n_splits, random_state)

    out = pd.DataFrame({"row_index": list(range(len(df))), "fold": folds})
    if id_column in df.columns:
        out.insert(0, id_column, df[id_column].astype(str).tolist())
    return out


def split_train_validation(df: pd.DataFrame, folds: pd.DataFrame, validation_fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "row_index" not in folds.columns or "fold" not in folds.columns:
        raise ValueError("folds must contain row_index and fold columns")
    fold_map = folds.set_index("row_index")["fold"]
    aligned = df.copy().reset_index(drop=True)
    fold_values = aligned.index.to_series().map(fold_map)
    if fold_values.isna().any():
        raise ValueError("fold assignment is missing rows from the dataframe")
    validation_mask = fold_values.astype(int) == int(validation_fold)
    return aligned[~validation_mask].reset_index(drop=True), aligned[validation_mask].reset_index(drop=True)

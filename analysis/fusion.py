
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from analysis.dataset import FeastDataset, as_list, normalise_cwe, normalise_cwes
from analysis.metrics import CWERelationshipMatcher, cwe_universe_for_metrics, metrics_to_dataframe


@dataclass(frozen=True)
class WeightedVotingStrategy:
    fire_metric: str
    silence_metric: str

    @property
    def name(self) -> str:
        return f"weighted_fire_{self.fire_metric}_silence_{self.silence_metric}"


DEFAULT_WEIGHTED_STRATEGIES = (
    WeightedVotingStrategy("ppv", "npv"),
    WeightedVotingStrategy("ppv", "sensitivity"),
    WeightedVotingStrategy("specificity", "npv"),
    WeightedVotingStrategy("specificity", "sensitivity"),
)


PERFORMANCE_METRICS = (
    "tp", "fp", "tn", "fn", "support", "positive_support", "negative_support",
    "precision", "recall", "specificity", "npv", "fpr", "fnr", "accuracy",
    "balanced_accuracy", "f1", "f2", "mcc", "roc_auc", "pr_auc",
)


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return float(value)


def _safe_div(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def _metrics_df(metrics: dict | pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(metrics, pd.DataFrame):
        return metrics.copy()
    if isinstance(metrics, (str, Path)):
        return pd.read_csv(metrics)
    return metrics_to_dataframe(metrics)


def _metric_value(row: dict, metric: str) -> float | None:
    if metric == "specificity":
        fpr = _safe_float(row.get("fpr"))
        return None if fpr is None else 1.0 - fpr
    if metric in {"sensitivity", "recall", "tpr"}:
        fnr = _safe_float(row.get("fnr"))
        return None if fnr is None else 1.0 - fnr
    return _safe_float(row.get(metric))


def _metric_lookup(metrics: dict | pd.DataFrame | str | Path) -> dict[tuple[str, str], dict]:
    df = _metrics_df(metrics)
    required = {"tool", "cwe", "supported"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metrics missing required column(s): {', '.join(sorted(missing))}")

    lookup: dict[tuple[str, str], dict] = {}
    for row in df.to_dict("records"):
        cwe = normalise_cwe(row.get("cwe"))
        tool = str(row.get("tool") or "")
        if not tool or cwe is None:
            continue
        lookup[(tool, cwe)] = row
    return lookup


def _dataset_df(dataset: FeastDataset | pd.DataFrame) -> pd.DataFrame:
    return dataset.to_pandas() if isinstance(dataset, FeastDataset) else dataset.copy()


def _tool_columns(dataset: FeastDataset | pd.DataFrame, tools: Iterable[str] | None = None) -> list[str]:
    if tools is not None:
        return list(tools)
    if isinstance(dataset, FeastDataset):
        return list(dataset.tool_columns)
    return [col for col in dataset.columns if col not in {"code", "language", "label", "cwes", "branch", "source", "code_hash", "sample_id"}]


def _tool_fires(row, tool: str, cwe: str) -> bool:
    if tool not in row:
        return False
    return cwe in normalise_cwes(as_list(row[tool]))


def _label_positive(row, cwe: str, matcher: CWERelationshipMatcher | None = None) -> bool:
    gt_cwes = normalise_cwes(as_list(row.get("cwes", [])))
    if matcher is None:
        return cwe in gt_cwes
    return matcher.any_vertical_match(cwe, gt_cwes)


def _supported(metric_row: dict | None) -> bool:
    if not metric_row:
        return False
    value = metric_row.get("supported")
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def cwe_universe(dataset: FeastDataset | pd.DataFrame, tools: Iterable[str] | None = None) -> list[str]:
    df = _dataset_df(dataset)
    return cwe_universe_for_metrics(df, _tool_columns(dataset, tools))


def weighted_vote_predictions(
    dataset: FeastDataset | pd.DataFrame,
    metrics: dict | pd.DataFrame | str | Path,
    strategy: WeightedVotingStrategy,
    tools: Iterable[str] | None = None,
    cwes: Iterable[str] | None = None,
    matcher: CWERelationshipMatcher | None = None,
    empty_score: float = 0.0,
) -> pd.DataFrame:
    df = _dataset_df(dataset).reset_index(drop=True)
    selected_tools = _tool_columns(dataset, tools)
    selected_cwes = sorted(normalise_cwes(cwes), key=lambda c: int(c.removeprefix("CWE-"))) if cwes is not None else cwe_universe(df, selected_tools)
    lookup = _metric_lookup(metrics)

    rows: list[dict] = []
    for row_index, row in df.iterrows():
        sample_id = row.get("sample_id", row_index)
        for cwe in selected_cwes:
            vuln_score = 0.0
            safe_score = 0.0
            tools_supported = 0
            tools_fired = 0
            tools_abstained = 0

            for tool in selected_tools:
                metric_row = lookup.get((tool, cwe))
                if not _supported(metric_row):
                    tools_abstained += 1
                    continue
                tools_supported += 1
                fired = _tool_fires(row, tool, cwe)
                if fired:
                    tools_fired += 1
                    weight = _metric_value(metric_row, strategy.fire_metric)
                    if weight is not None:
                        vuln_score += weight
                else:
                    weight = _metric_value(metric_row, strategy.silence_metric)
                    if weight is not None:
                        safe_score += weight

            total_score = vuln_score + safe_score
            score = empty_score if total_score == 0 else vuln_score / total_score
            rows.append({
                "sample_id": sample_id,
                "row_index": row_index,
                "cwe": cwe,
                "strategy": strategy.name,
                "prediction": bool(vuln_score >= safe_score and total_score > 0),
                "score": score,
                "vuln_score": vuln_score,
                "safe_score": safe_score,
                "tools_supported": tools_supported,
                "tools_fired": tools_fired,
                "tools_abstained": tools_abstained,
                "label": _label_positive(row, cwe, matcher),
            })
    return pd.DataFrame(rows)


def traditional_vote_predictions(
    dataset: FeastDataset | pd.DataFrame,
    tools: Iterable[str] | None = None,
    cwes: Iterable[str] | None = None,
    threshold: int = 2,
    matcher: CWERelationshipMatcher | None = None,
    metrics: dict | pd.DataFrame | str | Path | None = None,
    supported_only: bool = False,
) -> pd.DataFrame:
    df = _dataset_df(dataset).reset_index(drop=True)
    selected_tools = _tool_columns(dataset, tools)
    selected_cwes = sorted(normalise_cwes(cwes), key=lambda c: int(c.removeprefix("CWE-"))) if cwes is not None else cwe_universe(df, selected_tools)
    lookup = _metric_lookup(metrics) if metrics is not None else {}
    strategy_name = f"traditional_{threshold}_of_supported" if supported_only else f"traditional_{threshold}_of_{len(selected_tools)}"

    rows: list[dict] = []
    for row_index, row in df.iterrows():
        sample_id = row.get("sample_id", row_index)
        for cwe in selected_cwes:
            considered = []
            abstained = 0
            for tool in selected_tools:
                if supported_only and not _supported(lookup.get((tool, cwe))):
                    abstained += 1
                    continue
                considered.append(tool)
            fire_count = sum(1 for tool in considered if _tool_fires(row, tool, cwe))
            denom = len(considered)
            rows.append({
                "sample_id": sample_id,
                "row_index": row_index,
                "cwe": cwe,
                "strategy": strategy_name,
                "prediction": bool(fire_count >= threshold),
                "score": 0.0 if denom == 0 else fire_count / denom,
                "vuln_score": float(fire_count),
                "safe_score": float(max(denom - fire_count, 0)),
                "tools_supported": denom,
                "tools_fired": fire_count,
                "tools_abstained": abstained,
                "label": _label_positive(row, cwe, matcher),
            })
    return pd.DataFrame(rows)


def all_fusion_predictions(
    dataset: FeastDataset | pd.DataFrame,
    metrics: dict | pd.DataFrame | str | Path,
    tools: Iterable[str] | None = None,
    cwes: Iterable[str] | None = None,
    matcher: CWERelationshipMatcher | None = None,
    traditional_threshold: int = 2,
    include_supported_traditional: bool = True,
    strategies: Sequence[WeightedVotingStrategy] = DEFAULT_WEIGHTED_STRATEGIES,
) -> pd.DataFrame:
    frames = [
        traditional_vote_predictions(dataset, tools, cwes, traditional_threshold, matcher, supported_only=False),
    ]
    if include_supported_traditional:
        frames.append(traditional_vote_predictions(dataset, tools, cwes, traditional_threshold, matcher, metrics, supported_only=True))
    frames.extend(
        weighted_vote_predictions(dataset, metrics, strategy, tools, cwes, matcher)
        for strategy in strategies
    )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _roc_auc(labels: list[bool], scores: list[float]) -> float | None:
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    ordered = sorted(enumerate(scores), key=lambda item: item[1])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and ordered[j][1] == ordered[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ordered[k][0]] = avg_rank
        i = j

    pos_rank_sum = sum(rank for rank, label in zip(ranks, labels, strict=False) if label)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(labels: list[bool], scores: list[float]) -> float | None:
    n_pos = sum(labels)
    if n_pos == 0:
        return None
    ordered = sorted(zip(scores, labels, strict=False), key=lambda item: item[0], reverse=True)
    tp = 0
    precisions = []
    for rank, (_score, label) in enumerate(ordered, start=1):
        if label:
            tp += 1
            precisions.append(tp / rank)
    return sum(precisions) / n_pos


def _binary_metrics(labels: list[bool], predictions: list[bool], scores: list[float]) -> dict:
    tp = sum(label and pred for label, pred in zip(labels, predictions, strict=False))
    fp = sum((not label) and pred for label, pred in zip(labels, predictions, strict=False))
    tn = sum((not label) and (not pred) for label, pred in zip(labels, predictions, strict=False))
    fn = sum(label and (not pred) for label, pred in zip(labels, predictions, strict=False))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    npv = _safe_div(tn, tn + fn)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    accuracy = _safe_div(tp + tn, tp + fp + tn + fn)
    balanced_accuracy = None if recall is None or specificity is None else (recall + specificity) / 2

    def f_beta(beta: float) -> float | None:
        if precision is None or recall is None:
            return None
        beta2 = beta * beta
        den = beta2 * precision + recall
        return None if den == 0 else (1 + beta2) * precision * recall / den

    mcc_den = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = None if mcc_den == 0 else ((tp * tn) - (fp * fn)) / mcc_den

    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "support": int(len(labels)),
        "positive_support": int(sum(labels)),
        "negative_support": int(len(labels) - sum(labels)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "npv": npv,
        "fpr": fpr,
        "fnr": fnr,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "f1": f_beta(1.0),
        "f2": f_beta(2.0),
        "mcc": mcc,
        "roc_auc": _roc_auc(labels, scores),
        "pr_auc": _average_precision(labels, scores),
    }


def evaluate_predictions(predictions: pd.DataFrame, group_cols: Iterable[str] = ("strategy", "cwe")) -> pd.DataFrame:
    required = {"label", "prediction", "score"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required column(s): {', '.join(sorted(missing))}")

    group_cols = list(group_cols)
    rows: list[dict] = []
    grouped = [((), predictions)] if not group_cols else predictions.groupby(group_cols, dropna=False, sort=True)
    for key, group in grouped:
        if group.empty:
            continue
        labels = [bool(value) for value in group["label"].tolist()]
        preds = [bool(value) for value in group["prediction"].tolist()]
        scores = [float(value) for value in group["score"].fillna(0.0).tolist()]
        row = _binary_metrics(labels, preds, scores)
        if group_cols:
            if not isinstance(key, tuple):
                key = (key,)
            row = {**dict(zip(group_cols, key, strict=False)), **row}
        rows.append(row)
    return pd.DataFrame(rows)

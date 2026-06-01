
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from analysis.dataset import FeastDataset, as_list, normalise_cwe, normalise_cwes
from ingestion.cwe_navigator import CWENavigator

BASE_COLUMNS = {
    "code",
    "language",
    "label",
    "cwes",
    "branch",
    "source",
    "code_hash",
    "sample_id",
}


@dataclass
class CWERelationshipMatcher:
    navigator: CWENavigator

    def __post_init__(self) -> None:
        self._ancestor_cache: dict[str, set[str]] = {}
        self._descendant_cache: dict[str, set[str]] = {}
        self._closure_cache: dict[str, set[str]] = {}
        self._children_by_parent: dict[str, set[str]] = {}
        for child, parents in self.navigator.child_of.items():
            for parent in parents:
                self._children_by_parent.setdefault(parent, set()).add(child)

    @classmethod
    def from_xml(cls, cwe_xml_path: str | Path = "data/cwec_latest.xml") -> "CWERelationshipMatcher":
        return cls(CWENavigator(str(cwe_xml_path)))

    @staticmethod
    def _num(cwe: str) -> str | None:
        normalised = normalise_cwe(cwe)
        if normalised is None:
            return None
        return normalised.removeprefix("CWE-")

    def ancestors(self, cwe: str) -> set[str]:
        num = self._num(cwe)
        if num is None:
            return set()
        if num in self._ancestor_cache:
            return self._ancestor_cache[num]

        ancestors: set[str] = set()
        stack = list(self.navigator.child_of.get(num, []))
        while stack:
            parent = stack.pop()
            if parent in ancestors:
                continue
            ancestors.add(parent)
            stack.extend(self.navigator.child_of.get(parent, []))

        self._ancestor_cache[num] = ancestors
        return ancestors

    def childof_descendants(self, cwe: str) -> set[str]:
        """Return descendants reachable through ChildOf edges, excluding cousins."""
        num = self._num(cwe)
        if num is None:
            return set()
        if num in self._descendant_cache:
            return self._descendant_cache[num]

        descendants: set[str] = set()
        stack = list(self._children_by_parent.get(num, set()))
        while stack:
            child = stack.pop()
            if child in descendants:
                continue
            descendants.add(child)
            stack.extend(self._children_by_parent.get(child, set()))

        self._descendant_cache[num] = descendants
        return descendants

    def view_or_category_members(self, cwe: str) -> set[str]:
        """Return members under a MITRE View/Category target, flattened across depths."""
        num = self._num(cwe)
        if num is None:
            return set()
        descendants = self.navigator.get_descendants(num)
        if descendants.get("type") not in {"View", "Category"}:
            return set()

        members: set[str] = set()
        for by_depth in descendants.get("by_view", {}).values():
            for ids_at_depth in by_depth.values():
                members.update(ids_at_depth)
        return members

    def vertical_closure(self, cwe: str) -> set[str]:
        """Return CWE IDs covered vertically by a target CWE/View/Category.

        For a normal weakness this is the weakness plus all ChildOf descendants.
        For a MITRE View/Category, this is the view/category, every member in that
        view/category, and all ChildOf descendants of every member. The result is
        cached because metrics repeatedly validate the same target CWE.
        """
        num = self._num(cwe)
        if num is None:
            return set()
        if num in self._closure_cache:
            return self._closure_cache[num]

        closure = {num}
        closure.update(self.view_or_category_members(num))

        expanded = set(closure)
        for member in list(closure):
            expanded.update(self.childof_descendants(member))

        self._closure_cache[num] = expanded
        return expanded

    def is_vertical_match(self, a: str, b: str) -> bool:
        a_num = self._num(a)
        b_num = self._num(b)
        if a_num is None or b_num is None:
            return False
        return b_num in self.vertical_closure(a_num) or a_num in self.vertical_closure(b_num)

    def any_vertical_match(self, cwe: str, candidates: Iterable[str]) -> bool:
        target_closure = self.vertical_closure(cwe)
        if not target_closure:
            return False
        for candidate in candidates:
            candidate_num = self._num(candidate)
            if candidate_num is None:
                continue
            if candidate_num in target_closure or self._num(cwe) in self.vertical_closure(candidate_num):
                return True
        return False


def _safe_div(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def _row_tool_cwes(row, tool: str) -> set[str]:
    if tool not in row:
        return set()
    return normalise_cwes(as_list(row[tool]))


def _row_gt_cwes(row) -> set[str]:
    return normalise_cwes(as_list(row.get("cwes", [])))


def _cwe_sort_key(cwe: str) -> tuple[int, str]:
    return (int(cwe.removeprefix("CWE-")), cwe)


def tool_cwe_universe(df: pd.DataFrame, tool: str) -> list[str]:
    cwes: set[str] = set()
    if tool not in df.columns:
        return []
    for values in df[tool]:
        cwes.update(normalise_cwes(as_list(values)))
    return sorted(cwes, key=_cwe_sort_key)


def cwe_universe_for_metrics(
    df: pd.DataFrame,
    tools: Iterable[str],
    include_labels: bool = True,
    include_tool_outputs: bool = True,
) -> list[str]:
    """Return the CWE universe used for per-tool reliability estimation.

    The grid must include CWEs that a given tool never emitted, otherwise we
    cannot distinguish a reliable silence from an empirically unsupported CWE.
    """
    cwes: set[str] = set()
    if include_labels and "cwes" in df.columns:
        for values in df["cwes"]:
            cwes.update(normalise_cwes(as_list(values)))
    if include_tool_outputs:
        for tool in tools:
            if tool not in df.columns:
                continue
            for values in df[tool]:
                cwes.update(normalise_cwes(as_list(values)))
    return sorted(cwes, key=_cwe_sort_key)


def confusion_for_tool_cwe(
    df: pd.DataFrame,
    tool: str,
    cwe: str,
    matcher: CWERelationshipMatcher,
) -> dict[str, int | str]:
    target = normalise_cwe(cwe)
    if target is None:
        raise ValueError(f"Invalid CWE: {cwe!r}")

    tp = fp = tn = fn = 0
    for _, row in df.iterrows():
        tool_positive = target in _row_tool_cwes(row, tool)
        gt_positive = matcher.any_vertical_match(target, _row_gt_cwes(row))

        if tool_positive and gt_positive:
            tp += 1
        elif tool_positive and not gt_positive:
            fp += 1
        elif not tool_positive and gt_positive:
            fn += 1
        else:
            tn += 1

    return {"tool": tool, "cwe": target, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": len(df)}


def metrics_from_confusion(confusion: dict) -> dict:
    tp = int(confusion["tp"])
    fp = int(confusion["fp"])
    tn = int(confusion["tn"])
    fn = int(confusion["fn"])
    return {
        **confusion,
        "ppv": _safe_div(tp, tp + fp),
        "npv": _safe_div(tn, tn + fn),
        "fpr": _safe_div(fp, fp + tn),
        "fnr": _safe_div(fn, fn + tp),
        "supported": (tp + fp) > 0,
    }


def compute_tool_cwe_metrics(
    dataset: FeastDataset,
    cwe_xml_path: str | Path = "data/cwec_latest.xml",
    tools: Iterable[str] | None = None,
) -> dict:
    matcher = CWERelationshipMatcher.from_xml(cwe_xml_path)
    df = dataset.to_pandas()
    selected_tools = list(tools) if tools is not None else list(dataset.tool_columns)

    metrics: list[dict] = []
    if df.empty:
        languages: list[str] = []
    else:
        languages = sorted(df["language"].dropna().astype(str).unique())

    for language in languages:
        lang_df = df[df["language"].astype(str) == language].reset_index(drop=True)
        cwe_universe = cwe_universe_for_metrics(lang_df, selected_tools)
        for tool in selected_tools:
            if tool not in lang_df.columns:
                continue
            for cwe in cwe_universe:
                row = confusion_for_tool_cwe(lang_df, tool, cwe, matcher)
                metrics.append({"language": language, **metrics_from_confusion(row)})

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cwe_universe": "labels_and_tool_outputs",
            "matching": "A tool CWE is validated when it is vertically compatible with at least one ground-truth CWE: exact match, ancestor, descendant, or membership in a MITRE View/Category such as OWASP Top Ten 2021. Cousin relationships are excluded; View/Category targets expand to their member weaknesses and all ChildOf descendants.",
            "metric_definitions": {
                "ppv": "tp / (tp + fp)",
                "npv": "tn / (tn + fn)",
                "fpr": "fp / (fp + tn)",
                "fnr": "fn / (fn + tp)",
                "supported": "True when the tool produced at least one fire for the CWE in the calibration data: tp + fp > 0. Unsupported tool/CWE pairs should abstain during fusion.",
            },
            "input_rows": len(df),
            "languages": languages,
            "tools": selected_tools,
            "filters": list(dataset.history),
            "source_paths": [str(path) for path in dataset.source_paths],
            "cwe_xml_path": str(cwe_xml_path),
        },
        "metrics": metrics,
    }


def metrics_to_dataframe(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(payload.get("metrics", []))


def save_metrics_json(payload: dict, path: str | Path = "data/results/metrics.json") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def save_metrics_csv(payload: dict, path: str | Path = "data/results/metrics.csv") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_to_dataframe(payload).to_csv(path, index=False)
    return path


def save_metrics_outputs(
    payload: dict,
    json_path: str | Path = "data/results/metrics.json",
    csv_path: str | Path = "data/results/metrics.csv",
) -> tuple[Path, Path]:
    return save_metrics_json(payload, json_path), save_metrics_csv(payload, csv_path)

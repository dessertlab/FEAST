"""Fine-grained CWE ablation: quantifies how much of the family-level detection
signal is "exact/same-CWE" vs. "family-cousin" (same pillar_child family, but a
different raw CWE than the one in the ground truth).

Motivation: pillar_child canonicalisation (analysis/canonical.py) collapses raw
CWEs to the direct child of their CWE-1000 pillar so that tools and ground truth
can be matched at a common level of abstraction (see DESIGN.md and paper Sec. 3).
This is deliberate and well-motivated for cases where multiple tools describe the
*same* underlying weakness at different granularity (paper's Problem 2). But it
can also credit a tool with a family-level "hit" when it actually fired a
different, unrelated raw CWE that merely happens to share the same pillar
ancestor as the ground truth (e.g. CWE-79 XSS vs. CWE-89 SQLi both -> CWE-74).

This script measures, per (language, family), what fraction of family-level true
positives are:
  - "exact"  : the tool's fired raw CWE(s) for that family intersect the sample's
               ground-truth raw CWE(s) for that family (same specific weakness,
               possibly via a direct-child relation handled by normalise_cwe);
  - "cousin" : the family matches, but no raw CWE overlap -- the tool detected a
               different specific weakness that happens to share the pillar.

Run per language (requires data/enriched/<lang>.parquet and data/cwec_latest.xml,
neither of which ship in the git repo -- regenerate via `main.py enrich` after
`main.py download` + the Stage 1-3 notebooks):

    uv run python auxiliary/cwe_family_conflation_ablation.py --lang c_cpp
    uv run python auxiliary/cwe_family_conflation_ablation.py --lang all
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.canonical import CweCanonicalizer
from analysis.dataset import as_list, detect_tool_columns


def run_lang(lang: str, min_family_count: int | None, xml_path: Path) -> pd.DataFrame:
    enriched_path = ROOT / "data" / "enriched" / f"{lang}.parquet"
    if not enriched_path.exists():
        print(f"[{lang}] missing {enriched_path} -- run `main.py enrich` first. Skipping.")
        return pd.DataFrame()

    df = pd.read_parquet(enriched_path)
    tool_columns = detect_tool_columns(df)
    canon = CweCanonicalizer.from_xml(level="pillar_child", cwe_xml_path=str(xml_path))

    # restrict to the same family population used in the analysis, if a results
    # dir is available for this language (keeps the ablation comparable to the
    # published numbers instead of scoring on every family the raw data touches).
    results_family_path = ROOT / "data" / "results" / lang / "pillar_child" / "base" / "fusion_metrics_per_family.csv"
    allowed_families = None
    if results_family_path.exists():
        allowed_families = set(pd.read_csv(results_family_path)["family"].unique())

    rows = []
    for _, r in df.iterrows():
        if int(r["label"]) != 1:
            continue  # only vulnerable rows carry a ground-truth CWE to match against
        gt_raw = as_list(r["cwes"])
        gt_family_of = {c: canon.family(c) for c in gt_raw}
        gt_raw_by_family: dict[str, set[str]] = {}
        for c, fam in gt_family_of.items():
            if fam is None or (allowed_families is not None and fam not in allowed_families):
                continue
            gt_raw_by_family.setdefault(fam, set()).add(c)
        if not gt_raw_by_family:
            continue

        for tool in tool_columns:
            tool_raw = as_list(r[tool])
            if not tool_raw:
                continue
            tool_raw_by_family: dict[str, set[str]] = {}
            for c in tool_raw:
                fam = canon.family(c)
                if fam is None:
                    continue
                tool_raw_by_family.setdefault(fam, set()).add(c)

            for fam, gt_raw_set in gt_raw_by_family.items():
                tool_raw_set = tool_raw_by_family.get(fam)
                if not tool_raw_set:
                    continue  # tool did not fire this family -> not a TP, irrelevant here
                exact = bool(tool_raw_set & gt_raw_set)
                rows.append({"language": lang, "tool": tool, "family": fam, "exact": exact})

    if not rows:
        return pd.DataFrame()

    hits = pd.DataFrame(rows)
    per_family = (
        hits.groupby(["language", "family"])
        .agg(n_family_tp=("exact", "size"), n_exact=("exact", "sum"))
        .reset_index()
    )
    per_family["n_cousin"] = per_family["n_family_tp"] - per_family["n_exact"]
    per_family["pct_exact"] = (per_family["n_exact"] / per_family["n_family_tp"] * 100).round(1)
    return per_family.sort_values("n_family_tp", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="all", choices=["c_cpp", "java", "python", "all"])
    ap.add_argument("--min-family-count", type=int, default=None)
    ap.add_argument("--cwe-xml", default=str(ROOT / "data" / "cwec_latest.xml"))
    args = ap.parse_args()

    langs = ["c_cpp", "java", "python"] if args.lang == "all" else [args.lang]
    frames = [run_lang(lang, args.min_family_count, Path(args.cwe_xml)) for lang in langs]
    result = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) else pd.DataFrame()

    if result.empty:
        print("\nNo data available -- see the missing-file messages above.")
        return

    print("\n=== Per-family exact vs. cousin family-level TPs ===")
    print(result.to_string(index=False))

    print("\n=== Overall summary per language ===")
    overall = (
        result.groupby("language")
        .agg(n_family_tp=("n_family_tp", "sum"), n_exact=("n_exact", "sum"), n_cousin=("n_cousin", "sum"))
        .reset_index()
    )
    overall["pct_exact"] = (overall["n_exact"] / overall["n_family_tp"] * 100).round(1)
    print(overall.to_string(index=False))

    out_path = ROOT / "data" / "results" / "_cross_language" / "cwe_conflation_ablation.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved per-family breakdown to {out_path}")


if __name__ == "__main__":
    main()

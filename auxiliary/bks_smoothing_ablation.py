"""Compares BKS with the m-estimate smoothing actually used in the codebase (m=2.0)
against the raw maximum-likelihood formula as literally written in the paper's Eq. 11
(no smoothing).

Motivation: analysis/fusion/bks.py shrinks each fire-pattern's empirical vulnerability
rate toward the family's prior with an m-estimate:

    score = (positives + m * prior) / (total + m),   m = 2.0

The paper's Eq. 11 states the raw ratio ``positives / total`` with no such term. This
script runs both variants through the exact same 5-fold split, tools, and family set used
by the main pipeline (``analysis.experiment.prepare_canonical``), and reports:
  - support-weighted overall metrics per language for each variant, and
  - the Wilcoxon 95% CI of each variant against the 2ooN baseline -- the same test and
    pairing (per CWE family, N=families) behind the paper's Figure 4 --
so the two variants are directly comparable to each other and to the published numbers.

Both variants are scored at a *fixed* decision threshold tau, per language (see
NESTED_TAU below) -- not tau=0.5, and not each variant's own best tau. tau materially
changes BKS's precision/recall, so comparing the two variants at different thresholds
would confound the smoothing effect with the threshold effect. The values in NESTED_TAU
are the tau nested-selected for production BKS (m=2 smoothing) by
``analysis.experiment._select_tau_nested`` -- i.e. chosen on a calibration-internal
split, never on the held-out fold being scored -- read off the "bks_tau_0_X" row of each
language's committed ``mean_difference_ci_f1.csv`` after a full-tier run. Applying that
same tau to the raw variant isolates the smoothing effect; it does not claim to be the
raw formula's own optimum.

Raw-formula edge case not specified by the paper: a fire pattern never seen in
calibration makes the ratio positives/total undefined (0/0). Here it is treated the same
way every other strategy treats "no evidence" (score=0.0, prediction=False) -- a modelling
choice made for this ablation only, not a claim about what the paper intends.

Requires data/enriched/<lang>.parquet and data/cwec_latest.xml (neither ships in the git
repo).

    uv run python auxiliary/bks_smoothing_ablation.py --lang all
    uv run python auxiliary/bks_smoothing_ablation.py --lang java --tier full
    uv run python auxiliary/bks_smoothing_ablation.py --lang java --tau 0.5   # override
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import pandas as pd

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.aggregation import aggregate_fusion_metrics
from analysis.experiment import prepare_canonical
from analysis.folds import split_train_validation
from analysis.fusion import TIER_MIN_COUNT
from analysis.fusion.bks import bks_predictions
from analysis.fusion.common import build_fire_index, evidence_row, precompute_labels, sample_ids_of
from analysis.fusion.predictions import evaluate_predictions
from analysis.fusion.traditional import traditional_vote_predictions
from analysis.mean_difference_ci import mean_difference_ci

STRATEGY_SMOOTHED = "bks_smoothed_m2"
STRATEGY_RAW = "bks_raw_eq11"
BASELINE_THRESHOLD = 2  # matches the paper's 2ooN baseline for every language studied

# tau nested-selected for production BKS (m=2), full tier, --calibration sensitivity,
# specificity -- read from the "bks_tau_0_X" row of each language's
# data/results/<lang>/pillar_child/full/plots/mean_difference_ci_f1.csv after the
# 2026-09-21 re-run with analysis.experiment._select_tau_nested. See module docstring.
NESTED_TAU = {"c_cpp": 0.1, "java": 0.7, "python": 0.3}


def raw_bks_predictions(
    calibration_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    tools: Sequence[str],
    families: Sequence[str],
    labels: dict[tuple[int, str], bool],
    fire_index: tuple[dict[tuple[int, str], set[str]], list[set[str]]],
    tau: float = 0.5,
) -> pd.DataFrame:
    """BKS exactly as written in the paper's Eq. 11: raw empirical P(vulnerable | pattern),
    no m-estimate. A pattern unseen in calibration (total=0, undefined ratio) falls back to
    score=0.0 / prediction=False -- see module docstring."""
    cal_fire, _ = build_fire_index(calibration_df, tools)
    val_fire, _ = fire_index
    cal_labels = precompute_labels(calibration_df, families)
    sample_ids = sample_ids_of(validation_df)
    n_cal, n_val = len(calibration_df), len(validation_df)

    def pattern(fire, i, family):
        return tuple(family in fire[(i, t)] for t in tools)

    rows: list[dict] = []
    for family in families:
        pos: dict[tuple, int] = {}
        tot: dict[tuple, int] = {}
        for i in range(n_cal):
            p = pattern(cal_fire, i, family)
            tot[p] = tot.get(p, 0) + 1
            if cal_labels[(i, family)]:
                pos[p] = pos.get(p, 0) + 1

        for i in range(n_val):
            p = pattern(val_fire, i, family)
            total = tot.get(p, 0)
            score = (pos.get(p, 0) / total) if total > 0 else 0.0
            fired = sum(p)
            rows.append(evidence_row(
                sample_ids[i], i, family, STRATEGY_RAW,
                prediction=bool(score >= tau), score=float(score),
                vuln=float(score), safe=float(1.0 - score),
                k_supported=len(tools), k_fired=fired, k_abstained=0,
                label=labels[(i, family)],
            ))
    return pd.DataFrame(rows)


def run_lang(language: str, tier: str, n_splits: int, seed: int, tau: float) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    min_cwe_count = TIER_MIN_COUNT.get(tier, n_splits)
    try:
        prep = prepare_canonical(language, "pillar_child", n_splits=n_splits,
                                  min_cwe_count=min_cwe_count, seed=seed)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[{language}] {exc}")
        return None
    cdf, tools, families, folds = prep.cdf, prep.tools, prep.families, prep.folds
    if not families:
        print(f"[{language}] no families survive restriction at tier={tier}")
        return None
    print(f"[{language}] scoring both BKS variants at tau={tau}")

    frames = []
    for fold in sorted(folds["fold"].unique()):
        cal_df, val_df = split_train_validation(cdf, folds, validation_fold=fold)
        labels = precompute_labels(val_df, families)
        fire_index = build_fire_index(val_df, tools)

        smoothed = bks_predictions(cal_df, val_df, tools, families, labels, fire_index, tau=tau)
        smoothed["strategy"] = STRATEGY_SMOOTHED
        raw = raw_bks_predictions(cal_df, val_df, tools, families, labels, fire_index, tau=tau)
        baseline = traditional_vote_predictions(
            val_df, tools, families, BASELINE_THRESHOLD, labels, fire_index)

        for preds in (smoothed, raw, baseline):
            per_family = evaluate_predictions(preds, group_cols=("strategy", "family"))
            per_family.insert(0, "fold", fold)
            frames.append(per_family)

    per_family_per_fold = pd.concat(frames, ignore_index=True)
    per_family_mean, overall = aggregate_fusion_metrics(per_family_per_fold)
    overall.insert(0, "language", language)
    overall.insert(1, "tier", tier)
    overall.insert(2, "tau", tau)

    # Same test as Figure 4: Wilcoxon signed-rank on per-CWE-family fold-mean F1,
    # paired against the 2ooN baseline (N = number of surviving families).
    ci = mean_difference_ci(
        per_family_mean, metric="f1", tools=tools,
        preselected_strategies=[STRATEGY_SMOOTHED, STRATEGY_RAW],
    )
    ci.insert(0, "language", language)
    ci.insert(1, "tau", tau)

    out_dir = ROOT / "data" / "results" / "_cross_language"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_family_mean.to_csv(out_dir / f"bks_smoothing_{language}_per_family.csv", index=False)
    ci.to_csv(out_dir / f"bks_smoothing_{language}_ci.csv", index=False)

    return overall, ci


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", default="all", choices=["c_cpp", "java", "python", "all"])
    ap.add_argument("--tier", default="full", choices=["base", "medium", "full"],
                     help="Family-support floor; the paper's headline numbers use 'full' (>=100 GT occurrences)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=None,
                     help="Fixed decision threshold for both variants, overriding NESTED_TAU for every requested language")
    args = ap.parse_args()

    langs = ["c_cpp", "java", "python"] if args.lang == "all" else [args.lang]
    overall_rows, ci_rows = [], []
    for lang in langs:
        tau = args.tau if args.tau is not None else NESTED_TAU[lang]
        result = run_lang(lang, args.tier, args.n_splits, args.seed, tau)
        if result is None:
            continue
        overall, ci = result
        overall_rows.append(overall)
        ci_rows.append(ci)

    if not overall_rows:
        print("\nNo data available -- see the messages above.")
        return

    overall_all = pd.concat(overall_rows, ignore_index=True)
    ci_all = pd.concat(ci_rows, ignore_index=True)

    print("\n=== BKS smoothed (m=2) vs. raw (paper Eq. 11) -- support-weighted overall metrics ===")
    cols = ["language", "tau", "strategy", "f1_weighted", "precision_weighted", "recall_weighted",
            "mcc_weighted", "pr_auc_weighted", "n_families"]
    print(overall_all[[c for c in cols if c in overall_all.columns]].to_string(index=False))

    print("\n=== Wilcoxon 95% CI vs. 2ooN baseline (same test + pairing behind Figure 4) ===")
    ci_cols = ["language", "tau", "strategy", "baseline_strategy", "n_pairs", "mean_difference",
               "hodges_lehmann", "p_value", "ci_low", "ci_high", "status"]
    print(ci_all[[c for c in ci_cols if c in ci_all.columns]].to_string(index=False))

    out_dir = ROOT / "data" / "results" / "_cross_language"
    overall_all.to_csv(out_dir / "bks_smoothing_overall.csv", index=False)
    ci_all.to_csv(out_dir / "bks_smoothing_ci.csv", index=False)
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    main()

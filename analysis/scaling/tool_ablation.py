"""Analysis 2 — vary the number of tools *within* a language.

The cross-language comparison cannot tell "more tools help" apart from "C/C++ is just an
easier language", because across the three languages tool-count, dataset size and coverage
move together. This lens breaks that tie: it holds the language (and its data) fixed and
re-runs fusion on every tool subset of size k = 2..N. The result is a curve of fusion
performance against the number of tools, identified *within* a single language.

Each subset is a real fusion run (``run_language_level`` with the complement excluded),
sent to a scratch results dir so the canonical outputs are never touched; only the per-fold
metrics are read back and collapsed to a few support-weighted f1 numbers per subset.

Caveat surfaced in the output: restricting the tool set also changes which families survive
the support filter, so ``families_kept`` is recorded per subset — the family grid is not
identical across k.
"""

from __future__ import annotations

import itertools
import tempfile
from pathlib import Path

import pandas as pd

from analysis.canonical import CANONICAL_LEVEL
from analysis.scaling.common import (
    LEVEL,
    dataset_covariates,
    load_per_family_per_fold,
    results_dir,
    run_outcomes,
)


def _all_tools(language: str, tier: str, results_root: str | Path) -> list[str]:
    return list(dataset_covariates(language, tier, results_root=results_root)["tools"])


def _subsets(tools: list[str], sizes: list[int] | None, max_combos: int | None):
    """Yield (k, subset) for every requested size, optionally capping combinations per k."""
    sizes = sizes or list(range(2, len(tools) + 1))
    for k in sizes:
        if k < 1 or k > len(tools):
            continue
        combos = list(itertools.combinations(tools, k))
        if max_combos is not None and len(combos) > max_combos:
            combos = combos[:max_combos]
        for subset in combos:
            yield k, list(subset)


def ablate_tools(
    language: str,
    *,
    level: str = CANONICAL_LEVEL,
    tier: str = "full",
    sizes: list[int] | None = None,
    max_combos: int | None = None,
    n_splits: int = 5,
    threshold: int = 2,
    seed: int = 42,
    enriched_dir: str | Path = "data/enriched",
    results_root: str | Path = "data/results",
) -> pd.DataFrame:
    """Run the tool-count ablation for one language and write ``ablation/by_n_tools.csv``.

    ``sizes`` restricts which subset sizes to evaluate (default 2..N); ``max_combos`` caps how
    many combinations per size are run (handy to keep the C/C++ sweep affordable). Returns the
    long table, one row per evaluated subset.
    """
    from analysis.experiment import run_language_level

    tools = _all_tools(language, tier, results_root)
    if len(tools) < 2:
        print(f"[skip] {language}: only {len(tools)} tool(s), nothing to ablate")
        return pd.DataFrame()

    rows: list[dict] = []
    for k, subset in _subsets(tools, sizes, max_combos):
        exclude = [t for t in tools if t not in subset]
        with tempfile.TemporaryDirectory(prefix=f"ablate_{language}_") as scratch:
            summary = run_language_level(
                language, level, n_splits=n_splits, threshold=threshold, exclude=exclude,
                seed=seed, enriched_dir=enriched_dir, results_root=scratch, tier=tier,
                write_plots=False,
            )
            if not summary or "results_dir" not in summary:
                print(f"[warn] {language} {subset}: no result (likely no families survive)")
                continue
            pf = load_per_family_per_fold(language, tier, level=level, results_root=scratch)
        outcomes = run_outcomes(pf)
        rows.append({
            "language": language,
            "n_tools": k,
            "tools": "+".join(subset),
            "families_kept": summary.get("families_kept"),
            "gt_kept": summary.get("gt_occurrences_kept"),
            **outcomes,
        })
        print(f"  {language} k={k} [{'+'.join(subset)}]  "
              f"fusion_max={outcomes['fusion_f1_max']:.3f}  "
              f"traditional={outcomes['traditional_f1']:.3f}")

    table = pd.DataFrame(rows)
    out_dir = results_dir(language, tier, level=level, results_root=results_root) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "by_n_tools.csv"
    table.to_csv(csv_path, index=False)
    print(f"[write] {csv_path}")
    if not table.empty:
        _save_curve(table, out_dir / "by_n_tools", language)
    return table


def _save_curve(table: pd.DataFrame, out_stem: Path, language: str) -> None:
    """Fusion f1 (mean and ceiling over subsets) and the baseline against the tool count."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = table.groupby("n_tools").agg(
        fusion_mean=("fusion_f1_mean", "mean"),
        fusion_ceiling=("fusion_f1_max", "max"),
        traditional=("traditional_f1", "mean"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(agg["n_tools"], agg["fusion_ceiling"], "-^", color="#0072B2", label="best fusion (ceiling)")
    ax.plot(agg["n_tools"], agg["fusion_mean"], "-o", color="#56B4E9", label="fusion (mean over subsets)")
    ax.plot(agg["n_tools"], agg["traditional"], "-s", color="#D55E00", label="traditional baseline")
    # individual subsets as faint points to show spread at each k
    ax.scatter(table["n_tools"], table["fusion_f1_max"], s=14, color="#0072B2", alpha=0.25, zorder=2)
    ax.set_xlabel("number of tools in the ensemble")
    ax.set_ylabel("support-weighted f1")
    ax.set_title(f"Fusion performance vs tool count — {language}")
    ax.set_xticks(sorted(table["n_tools"].unique()))
    ax.grid(linestyle=":", linewidth=0.6, color="#bdbdbd", alpha=0.8)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out_stem.with_suffix('.png')}")

"""Analysis 1 — cross-language tool-coverage comparison.

The fastest test of the "Java loses because its tools collectively see too little" hunch.
``analysis.complementarity`` already computes, per language, the union (OR) recall and the
oracle headroom; this lens just runs it for every language and lays the verdicts side by
side against the dataset-level covariates (number of tools, findings, coverage), so the
ceiling each language's tool pool imposes is visible at a glance.

No fusion is re-run here — it reads enriched data only to recompute the coverage matrices.
Output: ``data/results/_cross_language/coverage.csv`` (+ a union-recall vs n_tools plot).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.canonical import CANONICAL_LEVEL
from analysis.complementarity import run_diagnostics
from analysis.scaling.common import LANGUAGES, dataset_covariates

CROSS_LANG_DIRNAME = "_cross_language"

# Verdict fields worth surfacing in the cross-language table (the rest stay in each
# language's own diagnostics/verdict.json).
_VERDICT_FIELDS = [
    "family_best_tool", "family_best_tool_recall", "family_or_recall", "family_recall_headroom",
    "detection_best_tool_recall", "detection_or_recall", "detection_recall_headroom",
    "auc_best_single", "auc_all_tools", "auc_gain_full_ensemble",
]


def compare_coverage(
    languages: list[str] | None = None,
    *,
    level: str = CANONICAL_LEVEL,
    tier: str = "full",
    n_splits: int = 5,
    seed: int = 42,
    enriched_dir: str | Path = "data/enriched",
    results_root: str | Path = "data/results",
) -> pd.DataFrame:
    """Build the cross-language coverage table and write it under ``_cross_language/``.

    For each language: covariates from the existing fusion ``config.json`` joined with the
    coverage verdict from ``run_diagnostics``.  Languages whose enriched data or results are
    missing are skipped with a note rather than aborting the whole comparison.
    """
    languages = languages or LANGUAGES
    rows: list[dict] = []
    for language in languages:
        try:
            cov = dataset_covariates(language, tier, level=level, results_root=results_root)
        except FileNotFoundError as exc:
            print(f"[skip] {language}: {exc}")
            continue
        try:
            verdict = run_diagnostics(
                language, level, n_splits=n_splits, seed=seed,
                enriched_dir=enriched_dir, results_root=results_root,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[skip] {language}: diagnostics failed: {exc}")
            continue
        if not verdict:
            continue
        row = {
            "language": language,
            "n_tools": cov["n_tools"],
            "n_samples": cov["n_samples"],
            "gt_kept": cov["gt_kept"],
            "kept_frac": cov["kept_frac"],
            "families_kept": cov["families_kept"],
            "tools": ",".join(cov["tools"]),
        }
        row.update({k: verdict.get(k) for k in _VERDICT_FIELDS})
        rows.append(row)

    table = pd.DataFrame(rows)
    out_dir = Path(results_root) / CROSS_LANG_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "coverage.csv"
    table.to_csv(csv_path, index=False)
    print(f"[write] {csv_path}")
    if not table.empty:
        _save_coverage_plot(table, out_dir / "union_recall_vs_n_tools")
    return table


def _save_coverage_plot(table: pd.DataFrame, out_stem: Path) -> None:
    """Union recall and oracle headroom against the number of tools, one point per language."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(table["n_tools"], table["family_or_recall"], s=70, color="#0072B2",
               edgecolor="#222222", zorder=3, label="union (OR) recall")
    ax.scatter(table["n_tools"], table["family_best_tool_recall"], s=70, color="#D55E00",
               edgecolor="#222222", marker="s", zorder=3, label="best single-tool recall")
    for _, r in table.iterrows():
        ax.annotate(str(r["language"]), (r["n_tools"], r["family_or_recall"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xlabel("number of tools")
    ax.set_ylabel("family recall")
    ax.set_title("Tool-pool coverage by language")
    ax.grid(linestyle=":", linewidth=0.6, color="#bdbdbd", alpha=0.8)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out_stem.with_suffix('.png')}")

"""Analysis 3 — vary the dataset size *within* a language (learning curve).

The companion to the tool ablation. If C/C++ wins because it has ~25x more findings rather
than because it is C/C++, then shrinking C/C++ to Java's size should erode its advantage.
This lens subsamples the enriched rows at a grid of fractions (each repeated under several
sampling seeds for stability) and re-runs the full fusion pipeline on each subsample,
producing a learning curve of fusion f1 against dataset size.

Subsampling is done by writing the reduced rows to a temporary ``enriched_dir`` and pointing
``run_language_level`` at it, so neither the canonical enriched data nor the canonical
results are touched. ``families_kept`` is recorded per point because smaller samples drop
more families below the support floor.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from analysis.canonical import CANONICAL_LEVEL
from analysis.scaling.common import (
    dataset_covariates,
    load_per_family_per_fold,
    results_dir,
    run_outcomes,
)

DEFAULT_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


def _enriched_path(language: str, enriched_dir: str | Path) -> Path:
    return Path(enriched_dir) / f"{language}.parquet"


def ablate_dataset_size(
    language: str,
    *,
    level: str = CANONICAL_LEVEL,
    tier: str = "full",
    fractions: list[float] | None = None,
    repeats: int = 3,
    n_splits: int = 5,
    threshold: int = 2,
    seed: int = 42,
    enriched_dir: str | Path = "data/enriched",
    results_root: str | Path = "data/results",
) -> pd.DataFrame:
    """Run the dataset-size learning curve for one language and write ``ablation/by_dataset_size.csv``.

    ``fractions`` are of the full enriched row count (default 0.1..1.0); each is sampled
    ``repeats`` times with different sampling seeds. The fusion fold seed stays fixed at
    ``seed`` so only the data subsample varies. Returns one row per (fraction, repeat).
    """
    from analysis.experiment import run_language_level

    src = _enriched_path(language, enriched_dir)
    if not src.exists():
        raise FileNotFoundError(f"missing enriched parquet: {src}")
    full = pd.read_parquet(src)
    n_total = len(full)
    fractions = fractions or DEFAULT_FRACTIONS

    rows: list[dict] = []
    for frac in fractions:
        n = max(1, round(frac * n_total))
        for rep in range(repeats):
            if frac >= 1.0 and rep > 0:
                continue  # the full set is identical across repeats
            sample = full.sample(n=n, random_state=seed + rep) if n < n_total else full
            with tempfile.TemporaryDirectory(prefix=f"datablate_{language}_") as scratch:
                tmp_enriched = Path(scratch) / "enriched"
                tmp_enriched.mkdir()
                sample.to_parquet(tmp_enriched / f"{language}.parquet")
                tmp_results = Path(scratch) / "results"
                try:
                    summary = run_language_level(
                        language, level, n_splits=n_splits, threshold=threshold, seed=seed,
                        enriched_dir=tmp_enriched, results_root=tmp_results, tier=tier,
                        write_plots=False,
                    )
                except (ValueError, FileNotFoundError) as exc:
                    print(f"[warn] {language} frac={frac} rep={rep}: {exc}")
                    continue
                if not summary or "results_dir" not in summary:
                    print(f"[warn] {language} frac={frac} rep={rep}: no families survive")
                    continue
                pf = load_per_family_per_fold(language, tier, level=level, results_root=tmp_results)
            outcomes = run_outcomes(pf)
            rows.append({
                "language": language,
                "fraction": frac,
                "repeat": rep,
                "n_samples": n,
                "families_kept": summary.get("families_kept"),
                "gt_kept": summary.get("gt_occurrences_kept"),
                **outcomes,
            })
            print(f"  {language} frac={frac:.2f} rep={rep} n={n}  "
                  f"fusion_max={outcomes['fusion_f1_max']:.3f}  "
                  f"traditional={outcomes['traditional_f1']:.3f}")

    table = pd.DataFrame(rows)
    out_dir = results_dir(language, tier, level=level, results_root=results_root) / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "by_dataset_size.csv"
    table.to_csv(csv_path, index=False)
    print(f"[write] {csv_path}")
    if not table.empty:
        _save_curve(table, out_dir / "by_dataset_size", language)
    return table


def _save_curve(table: pd.DataFrame, out_stem: Path, language: str) -> None:
    """Learning curve: fusion ceiling and baseline f1 against the number of samples."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = table.groupby("n_samples").agg(
        fusion_mean=("fusion_f1_max", "mean"),
        fusion_std=("fusion_f1_max", "std"),
        traditional_mean=("traditional_f1", "mean"),
    ).reset_index().sort_values("n_samples")

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.errorbar(agg["n_samples"], agg["fusion_mean"], yerr=agg["fusion_std"].fillna(0.0),
                fmt="-^", color="#0072B2", capsize=3, label="best fusion (ceiling)")
    ax.plot(agg["n_samples"], agg["traditional_mean"], "-s", color="#D55E00",
            label="traditional baseline")
    ax.scatter(table["n_samples"], table["fusion_f1_max"], s=14, color="#0072B2", alpha=0.25, zorder=2)
    ax.set_xlabel("number of samples")
    ax.set_ylabel("support-weighted f1")
    ax.set_title(f"Fusion performance vs dataset size — {language}")
    ax.grid(linestyle=":", linewidth=0.6, color="#bdbdbd", alpha=0.8)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {out_stem.with_suffix('.png')}")

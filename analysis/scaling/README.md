# `analysis/scaling/` — why fusion performance differs across languages

Java, Python and C/C++ get different fusion results (C/C++ best, Java worst). The natural
explanation — "it tracks the number of tools, the dataset size, and how much the tools
find" — cannot be tested with the three languages alone, because across those three the
candidate causes move together (C/C++ has more tools *and* more data *and* more coverage).
You cannot tell which one matters, or whether it is simply the language.

These four analyses break that tie by creating variation in one factor at a time, then a
final model puts the pieces together.

## The four analyses

| # | What it answers | Command | Output |
|---|---|---|---|
| 1 | Do the tools, *combined*, see enough in each language? | `scaling-coverage` | `data/results/_cross_language/coverage.csv` |
| 2 | Does adding tools help, *within* one language? | `scaling-tools` | `data/results/<lang>/pillar_child/<tier>/ablation/by_n_tools.csv` |
| 3 | Does the advantage survive shrinking the dataset? | `scaling-data` | `.../ablation/by_dataset_size.csv` |
| 4 | Putting it together: which factor actually drives it? | `scaling-meta` | `data/results/_cross_language/meta_regression/` |

Each writes a CSV and a plot. Nothing re-runs the ingestion pipeline; analyses 2–3 re-run
fusion on subsets via `experiment.run_language_level` into a scratch directory, so the
canonical results are never touched.

## How to run

```bash
uv sync                                            # once: installs statsmodels (analysis 4)

uv run python main.py scaling-coverage             # 1 — fast (minutes)
uv run python main.py scaling-tools --lang java    # 2 — SLOW: one fusion run per tool subset
uv run python main.py scaling-data  --lang all     # 3 — SLOW: one fusion run per data fraction
uv run python main.py scaling-meta                 # 4 — fast; consumes the outputs above
```

**Cost warning.** One fusion run takes several minutes. Analysis 2 runs *every* tool subset:
C/C++ has 6 tools → ~57 subsets → hours. Bound it with `--sizes` (which subset sizes) and
`--max-combos` (how many combinations per size), e.g.
`scaling-tools --lang c_cpp --sizes 2,4,6 --max-combos 4`. Every command has `--help`.

## How to read the results

- **`coverage.csv`** — per language. `family_or_recall` = what all tools catch together (the
  ceiling fusion can't beat); `family_recall_headroom` = room left for fusion. *Finding so far:
  Java's union recall is the highest, so "Java's tools see too little" does NOT explain its
  lower performance.*

- **`by_n_tools.png`** — x = number of tools. A rising curve means more tools genuinely help,
  shown within one language. Read the **absolute** `fusion_f1_max`, **not** the lift over the
  baseline: the traditional *K-of-N* baseline gets stricter as N grows (with 2 tools, "2-of-2"
  is an AND and collapses), so lift-over-baseline is not comparable across tool counts.

- **`by_dataset_size.png`** — x = number of samples. If C/C++ shrunk to Java's size drops to
  Java's level, dataset size is the driver; if it stays high, it is not.

- **`meta_regression/`** — `coef_tools.csv` / `coef_data.csv` give the within-language slope of
  each factor with a `p_value` (below 0.05 = trustworthy). `mixed_model_summary.txt` weighs all
  factors together with a random intercept on CWE family. The key question: once dataset size
  and coverage are in the model, does the *language* effect vanish? If yes, the gap is the
  conditions (data, tools), not the language — a result that would generalise to a fourth
  language.

A reminder running through all of it: **absolute f1** (C/C++ wins) and **lift over the
traditional baseline** (Java gains most, because its baseline is weak) are different questions.
The CSVs carry both: `fusion_f1_max` and `best_fusion_minus_traditional`.

## Files

- `common.py` — dataset covariates from each run's `config.json`; per-(family, fold) `delta_f1`
  and the `run_outcomes` summariser (reuses `analysis.mean_difference_ci`).
- `coverage_xlang.py` (1), `tool_ablation.py` (2), `data_ablation.py` (3), `meta_regression.py` (4).

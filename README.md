# FEAST

This repository contains the code for reproducing the experiments described in the paper _"CWE-Aware Calibration and Fusion of Open Source Static Analysis Tools for Vulnerability Detection"_ submitted to **Journal of Systems and Software**.

**Fusing Evidence Across Static Analysis Tools for CWE-Specific Vulnerability Detection**

Multi-language vulnerability dataset pipeline. Collects, normalises, and synthesises labelled code samples from 16 public sources across C/C++, Java, and Python, each annotated with CWE IDs from the MITRE catalogue. Then runs a full fusion experiment that calibrates per-tool reliability and compares every fusion strategy under cross-validation.

---

## Pipeline

```
Stage 0  Download         00_download_datasets.ipynb | main.py download
           Download all raw datasets  ->  data/raw/

Stage 1  Statistics       01_c_cpp.ipynb | 01_java.ipynb | 01_python.ipynb
           Per-language quality report  ->  outputs/stage1_<lang>_stats.xlsx

Stage 2  Synthesis        02_synthesis.ipynb | main.py synthesize
           CWE-filter + deduplicate    ->  data/processed/  and  data/merged/

Stage 3  Materialization  03_materialize.ipynb | main.py materialize
           Write source files          ->  data/materialized/

Stage 4  Tool enrichment  main.py enrich
           Collapse SAT JSON reports + merged parquets  ->  data/enriched/<lang>.parquet

Stage 5  Fusion           main.py fusion
           Calibrate reliability + compare fusion strategies  ->  data/results/<lang>/

Stage 5b Diagnostics      main.py diagnose
           Tool-complementarity analysis (oracle/diversity/CV)  ->  data/results/<lang>/diagnostics/

Stage 5c Scaling study    main.py scaling-coverage | scaling-tools | scaling-data | scaling-meta
           Why fusion performance differs across languages: cross-language coverage,
           within-language tool-count and dataset-size ablations, and a meta-regression
           that separates the dataset-level drivers from the language label
            ->  data/results/<lang>/.../ablation/  and  data/results/_cross_language/
```

---

## Repository structure

```
FEAST/
├── main.py                          # CLI (see Usage below)
├── sample_primevul.py               # One-off PrimeVul safe samples downsampler
├── pyproject.toml                   # Project configuration and dependencies
├── ingestion/                       # Stage 0–4 extraction library
│   ├── schema.py                    # FunctionSample dataclass
│   ├── cwe_navigator.py             # MITRE CWE XML parser and tree walker
│   ├── utils.py                     # shared helpers (CWE regex, NVD placeholders)
│   └── <source>.py                  # one extractor module per dataset
├── analysis/                        # Stage 5 fusion analysis library
│   ├── experiment.py                # end-to-end pipeline: canonicalise → fold → fuse → report
│   ├── calibration.py               # per-(tool, family) reliability metrics
│   ├── canonical.py                 # CWE → canonical family mapping (primary-path rule)
│   ├── folds.py                     # multilabel-stratified k-fold splitting
│   ├── aggregation.py               # fold-mean + support-weighted family aggregation
│   ├── complementarity.py           # oracle/diversity/CV tool-complementarity diagnostics
│   ├── mean_difference_ci.py        # non-parametric mean difference confidence intervals
│   ├── _mean_difference_ci.py       # old parametric paired t-test implementation (reference only)
│   ├── reporting.py                 # CSV + plot writers
│   ├── fusion/                      # voting and machine learning fusion logic
│   └── scaling/                     # scaling and meta-regression analysis logic
├── notebooks/
│   ├── 00_download_datasets.ipynb   # Stage 0 – download
│   ├── 01_c_cpp.ipynb               # Stage 1 – C/C++ statistics
│   ├── 01_java.ipynb                # Stage 1 – Java statistics
│   ├── 01_python.ipynb              # Stage 1 – Python statistics
│   ├── 02_synthesis.ipynb           # Stage 2 – process + merge
│   ├── 03_materialize.ipynb         # Stage 3 – write source files
│   └── 04_inspect.ipynb             # Stage 4 – (Obsolete/legacy inspection script)
├── auxiliary/                       # Helper analysis scripts and composition figures
│   ├── figures/                     # Dataset composition overview PDF and PNG
│   └── *.py                         # Scripts for count calculations and visualization
├── data/
│   ├── raw/                         # downloaded datasets (git-ignored)
│   ├── cwec_latest.xml              # MITRE CWE catalogue (auto-downloaded)
│   ├── processed/                   # per-dataset CWE-filtered parquets (git-ignored)
│   ├── merged/                      # final deduplicated parquets (git-ignored)
│   ├── materialized/                # individual source files for static analysis (git-ignored)
│   ├── SAT-reports/                 # static-analysis JSON reports (git-ignored)
│   ├── enriched/                    # per-language merged samples + tool columns (git-ignored)
│   └── results/                     # fusion experiment outputs
│       ├── _cross_language/         # cross-language tool-coverage and meta-regression data
│       │   ├── coverage.csv         # cross-language Union Recall data
│       │   └── meta_regression/     # mixed model summaries and stack data
│       ├── <lang>/
│       │   └── pillar_child/
│       │       ├── diagnostics/     # Stage 5b tool complementarity diagnostics
│       │       │   ├── reliability_heatmap.png  # heatmaps of tool reliability
│       │       │   └── *.csv        # conditional value, marginal contributions, diversity
│       │       └── <tier>/          # tier folders: base / medium / full
│       │           ├── config.json  # experiment parameters
│       │           ├── canonical_map.csv  # raw CWE -> canonical family
│       │           ├── folds.csv    # per-row fold assignment
│       │           ├── calibration_reliability.csv  # reliability metrics
│       │           ├── fusion_metrics_per_family.csv  # family metrics
│       │           ├── fusion_metrics_per_family_per_fold.csv  # detailed fold metrics
│       │           ├── fusion_metrics_overall.csv     # support-weighted metrics
│       │           ├── fusion_detection_overall.csv   # binary detection metrics
│       │           ├── fusion_tau_sweep.csv           # tau sweep sweep values
│       │           ├── fusion_operating_points.csv    # best-MCC tau per strategy
│       │           ├── plots/       # per-metric and per-family plots
│       │           └── ablation/    # tool-count and dataset-size ablation data
│       └── *.png / *.svg            # combined plots and overall calibration figures
├── outputs/
│   ├── stage1_c_cpp_stats.xlsx
│   ├── stage1_dataset_stats.xlsx    # overall dataset counts and breakdown
│   ├── stage1_java_stats.xlsx
│   └── stage1_python_stats.xlsx
├── tests/                           # Unit tests for ingestion and analysis modules
└── pyproject.toml
```
```

---

## Datasets

25 sources organised by language and branch.

### C/C++ — 11 sources

| Dataset       | Branch | Positives                                             | Negatives                       |
| ------------- | ------ | ----------------------------------------------------- | ------------------------------- |
| PrimeVul      | real   | `target=1`, single-function commit, CWE non-empty     | all `target=0` (explicit)       |
| ICVul         | real   | `before_change=True`, `fc_hash` in CVE-FC mapping     | none                            |
| CVEfixes(C)   | real   | C/C++ language, single-function commit, CWE non-empty | none                            |
| MegaVul       | real   | single-function commit, CWE non-empty                 | none                            |
| SecVulEval    | real   | `is_vulnerable=True`                                  | all `is_vulnerable=False`       |
| CrossVul(C)   | real   | `bad_*` files (vulnerable functions)                  | `good_*` files (fix-paired)     |
| SVEN(C)       | real   | `func_src_before`, CWE from `vul_type`                | `func_src_after` (fix-paired)   |
| Juliet(C)     | synth  | `*_bad.c` files                                       | `*_good*.c` files               |
| CASTLE        | synth  | `vulnerable=True`                                     | `vulnerable=False`              |
| FormAI        | ai     | `VULNERABLE` / ESBMC `error_type` mapped to CWE       | `NON-VULNERABLE` / safe samples |
| LLMSecEval(C) | ai     | `gen_scenario/*.c` (Copilot completions)              | none                            |

### Java — 5 sources

| Dataset         | Branch | Positives                                | Negatives                   |
| --------------- | ------ | ---------------------------------------- | --------------------------- |
| CVEfixes(Java)  | real   | Java language, single-function commit    | none                        |
| CrossVul(Java)  | real   | `bad_*` files                            | `good_*` files (fix-paired) |
| Juliet(Java)    | synth  | `*_bad.java` files                       | `*_good*.java` files        |
| OWASP(Java)     | synth  | `real vulnerability=true`                | `real vulnerability=false`  |
| CAPEC_LLM(Java) | ai     | LLM-generated snippets for CAPEC entries | none                        |

### Python — 9 sources

| Dataset           | Branch | Positives                                 | Negatives                     |
| ----------------- | ------ | ----------------------------------------- | ----------------------------- |
| CVEfixes(Python)  | real   | Python language, single-function commit   | none                          |
| PatchEval         | real   | `vul_func` where `language=Python`        | `fix_func` (fix-paired)       |
| CrossVul(Python)  | real   | `bad_*` files                             | `good_*` files (fix-paired)   |
| PyVul             | real   | `code_before`, CWE from commits map       | `code_after` (fix-paired)     |
| SVEN(Python)      | real   | `func_src_before`                         | `func_src_after` (fix-paired) |
| OWASP(Python)     | synth  | `real vulnerability=true`                 | `real vulnerability=false`    |
| LLMSecEval        | ai     | `gen_scenario/*.py` (Copilot completions) | `Secure/*.py` files           |
| SecurityEval      | ai     | all samples (vulnerable-only dataset)     | none                          |
| CAPEC_LLM(Python) | ai     | LLM-generated snippets for CAPEC entries  | none                          |

**Branch semantics:**

| Branch  | Meaning                                                                   |
| ------- | ------------------------------------------------------------------------- |
| `real`  | Functions extracted from actual CVE patches or real-world codebases       |
| `synth` | Template/rule-based synthesised code (Juliet test suite, OWASP Benchmark) |
| `ai`    | LLM-generated code (Copilot completions, ChatGPT-generated snippets)      |

---

## Output schema

Every extractor returns `list[FunctionSample]`:

```python
@dataclass
class FunctionSample:
    code: str        # function body
    cwes: list[str]  # CWE IDs (e.g. ["CWE-79"]); empty list for label=0
    label: int       # 1 = vulnerable,  0 = safe
    branch: str      # "real" | "synth" | "ai"
    language: str    # "C/C++" | "Java" | "Python"
    sample_id: str   # stable content-derived ID: SHA-256(norm(code))[:16]
```

`sample_id` is assigned at extraction time via a registry-level wrapper and is stable across runs (content-derived, not positional).

Parquet files produced by Stage 2 add two columns:

| Column      | Description                                                         |
| ----------- | ------------------------------------------------------------------- |
| `source`    | Dataset name (e.g. `"PyVul"`)                                       |
| `code_hash` | Full SHA-256 of normalised code (used for deduplication)            |
| `sample_id` | First 16 hex chars of `code_hash`; used as filename stem in Stage 3 |

---

## CWE classification

The pipeline classifies every CWE ID against `cwec_latest.xml` from MITRE:

| Type         | Meaning                                                |
| ------------ | ------------------------------------------------------ |
| `leaf`       | Most specific weakness; no children in MITRE hierarchy |
| `non-leaf`   | Parent or intermediate node (broad attribution)        |
| `category`   | MITRE organisational grouping, not a proper weakness   |
| `deprecated` | Superseded weakness (name starts with `DEPRECATED:`)   |
| `unknown`    | ID not found in `cwec_latest.xml`                      |

Stage 2 retains only `leaf` and `non-leaf` by default. The Excel reports colour-code CWE columns by type (green, amber, purple, pink, gray).

---

## Setup

```bash
# recommended: install with uv
uv sync --all-extras

# alternative: pip
pip install -e ".[dev]"
```

Requires Python >= 3.10.

```bash
# run tests
uv run pytest
```

---

## Notebooks

Run in order. All notebooks are idempotent.

### `00_download_datasets.ipynb` — Stage 0

Downloads all 17 raw datasets to `data/raw/`. Each section skips if the target path already exists. Run this **once** before any other notebook. Equivalent to `main.py download`.

Two datasets require manual download from Zenodo and cannot be fetched programmatically:

- **CrossVul** — place `crossvul.zip` at `data/raw/crossvul.zip`
- **LLMSecEval (vulnerable)** — place `copilot-cwe-scenarios-dataset.zip` at `data/raw/copilot-cwe-scenarios-dataset.zip` (Zenodo record 5225651)

### `01_c_cpp.ipynb` / `01_java.ipynb` / `01_python.ipynb` — Stage 1

Per-language quality analysis. For each source dataset:

1. Extracts `FunctionSample` collections via the `ingestion/` library
2. Computes total, vulnerable, and safe sample counts
3. Builds a per-CWE distribution matrix
4. Classifies every CWE ID by MITRE type

**Output:** `outputs/stage1_{c_cpp,java,python}_stats.xlsx`

Each workbook contains:

- One sheet per branch: `{lang}_Real`, `{lang}_Synth`, `{lang}_AI`
- A `Filters` sheet documenting extraction methodology for every source
- A `Legend` sheet explaining CWE colour coding

### `02_synthesis.ipynb` — Stage 2

For each language:

1. **Process** — applies the CWE filter (leaf + non-leaf only by default) to every source; saves one parquet per dataset to `data/processed/<lang>/`
2. **Merge** — concatenates all processed datasets, deduplicates by SHA-256 of normalised code (higher-quality branch wins: `real > synth > ai`); saves the result to `data/merged/<lang>_merged.parquet`

### `03_materialize.ipynb` — Stage 3

For each language reads `data/merged/<lang>_merged.parquet` and writes:

- One source file per sample: `data/materialized/<lang>/<dataset>/<sample_id>.<ext>`
- A lookup index: `data/materialized/<lang>/index.parquet` mapping `sample_id` to `source`, `label`, `cwes`, `branch`, and `code_hash`

Files are skipped if they already exist (set `OVERWRITE = True` to force re-write). Existing files are never deleted.

> **Java note:** Samples are function bodies extracted from CVE patches, not compilable top-level classes. Static analysis tools that require a full compilation unit will need an additional wrapping step.

### `main.py enrich` — Stage 4

Reads SAT report JSON files from `data/SAT-reports/` and merged parquet files from `data/merged/`, then writes one enriched parquet per language to `data/enriched/<lang>.parquet` (`c_cpp.parquet`, `java.parquet`, `python.parquet`). The output preserves the merged dataset columns and adds one list-valued column per static-analysis tool that actually ran for that language. Each tool column contains the unique CWE IDs reported by that tool for the sample, or an empty list when the tool reported no CWE for that sample.

Matching primarily uses the materialized path shape `<dataset>/<sample_id>.<ext>` and falls back to `sample_id` within the same language when older report paths differ slightly. The enrich step reads the per-sample `runs[].tools` section first so tools with zero findings are still represented, and then folds in `findings` as additional evidence.

---

## CLI

`main.py` exposes the full pipeline from the command line.

### `download` — fetch all datasets (Stage 0)

```bash
uv run python main.py download
```

Downloads all 17 datasets to `data/raw/`. Idempotent: already-present paths are skipped. Two datasets require manual download from Zenodo; the command prints instructions for these when they are missing:

| Dataset                 | File to place in `data/raw/`                                |
| ----------------------- | ----------------------------------------------------------- |
| CrossVul                | `crossvul.zip`                                              |
| LLMSecEval (vulnerable) | `copilot-cwe-scenarios-dataset.zip` (Zenodo record 5225651) |

### `list` — show all available sources

```bash
uv run python main.py list
```

### `synthesize` — build processed and merged parquets

```bash
uv run python main.py synthesize [OPTIONS]
```

| Option                     | Default         | Description                                                           |
| -------------------------- | --------------- | --------------------------------------------------------------------- |
| `--lang LANG`              | `all`           | Language to process: `c`, `java`, `python`, or `all`                  |
| `--sources SRC1,SRC2,...`  | all available   | Comma-separated dataset names to include                              |
| `--cwe-types TYPES`        | `leaf,non-leaf` | CWE node types to retain for vulnerable samples                       |
| `--cwes CWE-79,CWE-89,...` | all             | Explicit whitelist of CWE IDs                                         |
| `--min-cwe-count N`        | `1` (off)       | Drop CWEs with fewer than N vulnerable samples after merge            |
| `--branches BRANCHES`      | `all`           | Source branches to include: `real`, `synth`, `ai`, or comma-separated |
| `--data-dir DIR`           | `data/raw/`     | Override the raw data directory                                       |

`--cwe-types`, `--cwes`, and `--min-cwe-count` compose independently: a vulnerable sample is kept only if it satisfies all active filters simultaneously.

**Examples:**

```bash
# full pipeline, all languages, default filters
uv run python main.py synthesize

# Python only
uv run python main.py synthesize --lang python

# select specific sources (exact names or slug-style both accepted)
uv run python main.py synthesize --lang python --sources "CVEfixes(Python),PyVul,PatchEval"
uv run python main.py synthesize --lang c --sources primevul,icvul,secvuleval

# keep only the most specific CWEs
uv run python main.py synthesize --cwe-types leaf

# target a specific set of CWEs
uv run python main.py synthesize --cwes CWE-79,CWE-89,CWE-22,CWE-78

# drop CWEs that appear in fewer than 20 vulnerable samples (after merge)
uv run python main.py synthesize --min-cwe-count 20

# exclude AI-generated data
uv run python main.py synthesize --branches real,synth

# compose multiple filters
uv run python main.py synthesize \
    --lang python \
    --branches real \
    --cwe-types leaf \
    --min-cwe-count 10
```

Output is written to `data/processed/<lang>/` (one parquet per source) and `data/merged/<lang>_merged.parquet` (final deduplicated dataset). Both directories are created automatically if they do not exist.

### `materialize` — write source files for static analysis

```bash
uv run python main.py materialize [OPTIONS]
```

| Option        | Default | Description                                              |
| ------------- | ------- | -------------------------------------------------------- |
| `--lang LANG` | `all`   | Language to materialize: `c`, `java`, `python`, or `all` |
| `--overwrite` | off     | Re-write files that already exist                        |

Reads `data/merged/<lang>_merged.parquet` and writes one file per sample to `data/materialized/<lang>/<dataset>/<sample_id>.<ext>`. Also writes a per-language `index.parquet` lookup table. Existing files are skipped unless `--overwrite` is set.

**Examples:**

```bash
# materialize all languages
uv run python main.py materialize

# Python only
uv run python main.py materialize --lang python

# force re-write all existing files
uv run python main.py materialize --overwrite
```

### `enrich` — add SAT results to merged samples

```bash
uv run python main.py enrich [OPTIONS]
```

| Option              | Default             | Description                                     |
| ------------------- | ------------------- | ----------------------------------------------- |
| `--merged-dir DIR`  | `data/merged/`      | Directory containing merged parquet files       |
| `--reports-dir DIR` | `data/SAT-reports/` | Directory containing SAT report JSON files      |
| `--out-dir DIR`     | `data/enriched/`    | Output directory for per-language parquet files |

Reads every `*.parquet` in `data/merged/` and every language-named `*.json` in `data/SAT-reports/`, then writes one parquet per language with one additional list-valued column per tool that ran for that language.

**Examples:**

```bash
# default enrichment
uv run python main.py enrich

# custom output directory
uv run python main.py enrich --out-dir data/enriched_experiment
```

### `fusion` — calibrate reliability and compare fusion strategies (Stage 5)

```bash
uv run python main.py fusion [OPTIONS]
# aliases: fuse, f, analyze
```

Canonicalises CWE IDs to the direct children of CWE-1000 pillars (primary-path rule), calibrates per-(tool, family) reliability with exact matching, and runs every fusion strategy under stratified k-fold cross-validation. Results are written to `data/results/<lang>/pillar_child/<tier>/`.

| Option                 | Default         | Description                                                                                   |
| ---------------------- | --------------- | --------------------------------------------------------------------------------------------- |
| `--lang LANG`          | `all`           | Language: `c`, `java`, `python`, or `all`                                                     |
| `--exclude TOOL1,...`  | none            | Comma-separated tools to drop from the ensemble                                               |
| `--n-splits N`         | `5`             | Number of cross-validation folds                                                              |
| `--tier TIER`          | `base`          | Analysis tier (see below)                                                                     |
| `--min-cwe-count M`    | set by `--tier` | Override the tier's family support floor                                                      |
| `--threshold K`        | `2`             | K for the traditional K-of-N voting baseline                                                  |
| `--calibration M1,...` | all pairs       | Calibration metric pairs to include: `ppv`, `npv`, `sensitivity`, `specificity`, `fpr`, `fnr` |
| `--taumin T`           | `0.1`           | Lower bound of the τ sweep grid                                                               |
| `--taumax T`           | `0.9`           | Upper bound of the τ sweep grid                                                               |
| `--seed S`             | `42`            | Random seed for fold assignment                                                               |
| `--pvalues`            | off             | Show p-values above whiskers/error bars in the plot                                           |

#### Analysis tiers

The `--tier` flag controls two things simultaneously: the minimum number of ground-truth occurrences required to include a CWE family, and which ML-based strategies are activated.

| Tier     | Family support floor   | ML strategies added                               | Use when                                             |
| -------- | ---------------------- | ------------------------------------------------- | ---------------------------------------------------- |
| `base`   | ≥ n_splits (default 5) | none                                              | Maximum CWE coverage; existing strategies only       |
| `medium` | ≥ 30                   | Decision Tree                                     | Balanced coverage + one ML baseline                  |
| `full`   | ≥ 100                  | Decision Tree + Random Forest + Gradient Boosting | Highest-confidence families only; full ML comparison |

The ML classifiers use tool fire indicators (one binary feature per tool) as input and are trained on the calibration split of each fold. Class imbalance is handled via `class_weight='balanced'` (DT, RF) or inverse-frequency sample weights (GB), replacing SMOTE which is inapplicable on binary feature spaces. Hyperparameters are adapted from D'Abruzzo Pereira et al. (2024) to the 4-binary-feature regime of FEAST (see `analysis/fusion/ml.py`).

`--min-cwe-count` overrides the tier's support floor if you need a custom threshold. The tier still determines which ML strategies are included.

**Outputs** written to `data/results/<lang>/pillar_child/<tier>/`:

| File                                     | Description                                                                         |
| ---------------------------------------- | ----------------------------------------------------------------------------------- |
| `config.json`                            | All experiment parameters                                                           |
| `canonical_map.csv`                      | Raw CWE → canonical family mapping                                                  |
| `folds.csv`                              | Per-row fold assignment                                                             |
| `calibration_reliability.csv`            | Per-(tool, family, fold): TP/FP/TN/FN, PPV, NPV, FPR, FNR, sensitivity, specificity |
| `fusion_metrics_per_family_per_fold.csv` | Per-(strategy, family, fold) metrics for each fold                                  |
| `fusion_metrics_per_family.csv`          | Per-(strategy, family) mean metrics over folds                                      |
| `fusion_metrics_overall.csv`             | Support-weighted aggregate per strategy                                             |
| `fusion_detection_overall.csv`           | Vuln/safe binary detection metrics per strategy                                     |
| `fusion_tau_sweep.csv`                   | All (base_strategy, τ) combinations                                                 |
| `fusion_operating_points.csv`            | Best-MCC τ per strategy                                                             |
| `plots/`                                 | Per-metric bar charts, best variants, and per-family plots                          |
| `ablation/`                              | Tool-count and dataset-size ablation data (if scaling runs were executed)           |

**Fusion strategies** included in every tier:

| Strategy                                   | Type                                                |
| ------------------------------------------ | --------------------------------------------------- |
| `tool:<name>`                              | Single-tool baseline (one per tool)                 |
| `or_1_of_N`                                | OR of all tools                                     |
| `traditional_K_of_N`                       | K-of-N majority vote                                |
| `weighted_fire_<fire>_silence_<silence>`   | Reliability-weighted voting (3 metric pairs)        |
| `dst_<rule>_fire_<fire>_silence_<silence>` | Dempster-Shafer (Dempster, PCR6, Yager × 3 pairs)   |
| `naive_bayes`                              | Naive Bayes over log-likelihood ratios              |
| `bks`                                      | Behavior-Knowledge Space (empirical pattern lookup) |
| `logistic_regression`                      | Per-family logistic regression on fire indicators   |
| `logistic_interactions`                    | Same + pairwise tool-interaction features           |

**Examples:**

```bash
# base tier: all CWE families, existing strategies
uv run python main.py fusion --lang python

# medium tier: families with ≥ 30 samples, adds Decision Tree
uv run python main.py fusion --lang python --tier medium

# full tier: families with ≥ 100 samples, adds DT + RF + GB
uv run python main.py fusion --lang python --tier full

# full tier, all languages
uv run python main.py fusion --tier full

# exclude one tool
uv run python main.py fusion --lang python --exclude pylint

# custom min-cwe-count (overrides the tier floor)
uv run python main.py fusion --lang python --tier full --min-cwe-count 50

# restrict calibration metrics
uv run python main.py fusion --lang python --calibration ppv,npv

# run base tier and show p-values on plot
uv run python main.py fusion --lang python --tier base --pvalues
```

### `diagnose` — tool-complementarity diagnostics (Stage 5b)

```bash
uv run python main.py diagnose [OPTIONS]
# alias: diag
```

Runs tool-complementarity diagnostics on the same canonicalised data used by `fusion`: oracle/coverage headroom, error diversity, per-(tool, family) reliability heatmap, and cross-validated marginal contribution and conditional value per tool. Outputs are written to `data/results/<lang>/pillar_child/<tier>/diagnostics/`.

| Option                | Default      | Description                               |
| --------------------- | ------------ | ----------------------------------------- |
| `--lang LANG`         | `all`        | Language: `c`, `java`, `python`, or `all` |
| `--tier TIER`         | `base`       | Tier folder to read results from          |
| `--exclude TOOL1,...` | none         | Comma-separated tools to exclude          |
| `--n-splits N`        | `5`          | Number of cross-validation folds          |
| `--min-cwe-count M`   | `= n_splits` | Min GT occurrences per family             |
| `--seed S`            | `42`         | Random seed                               |

**Examples:**

```bash
# diagnostics for all languages
uv run python main.py diagnose

# Python only
uv run python main.py diagnose --lang python

# exclude a tool before computing marginal contributions
uv run python main.py diagnose --lang python --exclude devaic
```

### `plots` — regenerate plots from existing CSV results

```bash
uv run python main.py plots [OPTIONS]
# aliases: replot, plot
```

Regenerates all plots and the combined CI reports using the existing fusion output CSVs without repeating the cross-validation or fusion pipeline. Outputs are saved to `data/results/<lang>/pillar_child/<tier>/plots/` (as well as combined figures under `data/results/`).

| Option               | Default        | Description                                                       |
| -------------------- | -------------- | ----------------------------------------------------------------- |
| `--lang LANG`        | `all`          | Language: `c`, `java`, `python`, or `all`                         |
| `--tier TIER`        | `base`         | Tier folder containing the source CSV files: `base, medium, full` |
| `--results-dir DIR`  | `data/results` | Root results directory                                            |
| `--pvalues`          | off            | Show p-values above whiskers/error bars in the plot               |

**Examples:**

```bash
# replot base tier python results
uv run python main.py plots --lang python --tier base

# replot full tier for all languages and overlay pvalues
uv run python main.py plots --tier full --pvalues
```

### `scaling-coverage` — cross-language tool coverage analysis (Stage 5c)

```bash
uv run python main.py scaling-coverage [OPTIONS]
# alias: scov
```

Lays each language's union (OR) recall and oracle headroom side by side against its dataset-level covariates to see whether performance differences correlate with collective tool coverage. Writes to `data/results/_cross_language/coverage.csv`.

| Option          | Default | Description                                              |
| --------------- | ------- | -------------------------------------------------------- |
| `--lang LANG`   | `all`   | Language to analyze: `c`, `java`, `python`, or `all`     |
| `--tier TIER`   | `full`  | Tier subdirectory to read covariates from                |
| `--n-splits N`  | `5`     | Number of cross-validation folds                         |
| `--seed S`      | `42`    | Random seed for fold assignment                          |

**Examples:**

```bash
uv run python main.py scaling-coverage --tier full
```

### `scaling-tools` — tool count ablation (Stage 5c)

```bash
uv run python main.py scaling-tools [OPTIONS]
# alias: stools
```

Subsamples and re-runs the fusion pipeline on every tool subset combination of size $k = 2..N$ within a language, measuring fusion F1 against tool pool size. Writes outputs to `data/results/<lang>/pillar_child/<tier>/ablation/by_n_tools.csv`.

| Option           | Default | Description                                                               |
| ---------------- | ------- | ------------------------------------------------------------------------- |
| `--lang LANG`    | `all`   | Language: `c`, `java`, `python`, or `all`                                 |
| `--tier TIER`    | `full`  | Analysis tier to run scaling on                                           |
| `--sizes K1,K2`  | `2..N`  | Comma-separated subset sizes to evaluate                                  |
| `--max-combos M` | all     | Cap of combinations evaluated per size                                    |
| `--n-splits N`   | `5`     | Number of cross-validation folds                                          |
| `--threshold K`  | `2`     | K for the traditional K-of-N voting baseline                              |
| `--seed S`       | `42`    | Random seed                                                               |

**Examples:**

```bash
# run tool ablation on Python full tier
uv run python main.py scaling-tools --lang python --tier full
```

### `scaling-data` — dataset size ablation learning curves (Stage 5c)

```bash
uv run python main.py scaling-data [OPTIONS]
# alias: sdata
```

Subsamples the enriched samples across a grid of fractions and re-runs the fusion pipeline to see how performance scales with dataset size. Writes outputs to `data/results/<lang>/pillar_child/<tier>/ablation/by_dataset_size.csv`.

| Option              | Default                | Description                                                |
| ------------------- | ---------------------- | ---------------------------------------------------------- |
| `--lang LANG`       | `all`                  | Language: `c`, `java`, `python`, or `all`                  |
| `--tier TIER`       | `full`                 | Analysis tier to run scaling on                            |
| `--fractions F1,F2` | `0.1,0.25,0.5,0.75,1.0` | Row fractions of the dataset to evaluate                   |
| `--repeats R`       | `3`                    | Subsamples evaluated per fraction (different seeds)        |
| `--n-splits N`      | `5`                    | Number of cross-validation folds                           |
| `--threshold K`     | `2`                    | K for the traditional K-of-N voting baseline               |
| `--seed S`          | `42`                   | Base random seed                                           |

**Examples:**

```bash
# run dataset size ablation for python
uv run python main.py scaling-data --lang python --tier full
```

### `scaling-meta` — cross-language meta-regression (Stage 5c)

```bash
uv run python main.py scaling-meta [OPTIONS]
# alias: smeta
```

Stacks every (family, fold) fusion lift with its dataset-level covariates and fits a linear mixed-effects model (with a random intercept on CWE family) alongside within-language ablation regressions. Writes outputs to `data/results/_cross_language/meta_regression/`.

| Option        | Default | Description                                                    |
| ------------- | ------- | -------------------------------------------------------------- |
| `--lang LANG` | `all`   | Languages to include: `all` or a single language slug          |
| `--tier TIER` | `full`  | Tier subdirectory to read results from                         |

**Examples:**

```bash
uv run python main.py scaling-meta --tier full
```

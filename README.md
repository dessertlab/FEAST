# FEAST

**Fusing Evidence Across Static Analysis Tools for CWE-Specific Vulnerability Detection**

FEAST is a research pipeline that collects labeled C/C++ function samples from multiple vulnerability datasets, runs static analysis tools on them, and fuses the resulting per-CWE performance signals using Dempster-Shafer Theory (DST) to produce a combined confidence estimate for each weakness class.

---

## Pipeline overview

```
Stage 1 -- Dataset ingestion
  Download raw datasets  ->  Extract & normalise  ->  CWE quality report

Stage 2 -- Tool execution        (planned)
  Run static analysis tools on each FunctionSample collection

Stage 3 -- Per-CWE metric computation  (planned)
  Compute TP/FP/TN/FN per CWE per tool on each collection

Stage 4 -- DST fusion            (planned)
  Combine per-CWE evidence across tools into a single confidence index
```

Each stage produces self-contained outputs. Collections are kept separate across all stages; no cross-dataset merging is performed.

---

## Repository structure

```
FEAST/
|-- ingestion/                  # Stage 1 -- extraction library
|   |-- schema.py               # FunctionSample dataclass
|   |-- primevul.py             # PrimeVul extractor
|   |-- icvul.py                # ICVul extractor
|   |-- cvefixes.py             # CVEfixes extractor
|   |-- megavul.py              # MegaVul extractor
|   |-- secvuleval.py           # SecVulEval extractor
|   +-- cwe_navigator.py        # MITRE CWE XML parser and tree walker
|
|-- notebooks/
|   |-- 00_download_datasets.ipynb   # download all raw datasets to data/raw/
|   +-- 01_datasets_composition.ipynb  # extract, classify CWEs, write xlsx report
|
|-- tests/ingestion/            # unit tests for all extractors (41 tests)
|
|-- data/
|   |-- raw/                    # raw dataset files (git-ignored)
|   +-- cwec_latest.xml         # MITRE CWE catalogue (downloaded by notebook 01)
|
|-- outputs/
|   +-- stage1_dataset_stats.xlsx   # Stage 1 deliverable
|
+-- pyproject.toml
```

---

## Stage 1 -- Dataset ingestion

### Goal

Normalise five heterogeneous vulnerability datasets into a single schema so that downstream tool execution and metric computation can treat them uniformly.

### Output schema

Every extractor returns `list[FunctionSample]`:

```python
@dataclass
class FunctionSample:
    code: str          # C/C++ function body (vulnerable version)
    cwes: list[str]    # CWE IDs attributed to this function; [] for label=0
    label: int         # 1 = vulnerable,  0 = safe
```

`cwes` is a list because a single CVE can be tagged with multiple CWEs. It is kept as-is rather than exploded into one record per CWE to avoid inflating counts and duplicating function bodies. Safe samples always have `cwes = []`: attributing a CWE to a function explicitly labeled safe would be methodologically incorrect.

### Quality filters applied by all extractors

| Filter | Rationale |
|--------|-----------|
| CWE non-empty | Samples without a CWE attribution cannot contribute to per-CWE metrics |
| NVD placeholder CWEs dropped (`NVD-CWE-noinfo`, `NVD-CWE-Other`) | These are database-level placeholders, not actual weakness identifiers |
| Concatenated IDs split (`CWE-20CWE-190` -> `["CWE-20","CWE-190"]`) | Some sources store multiple CWEs concatenated; split by regex `CWE-\d+` |
| Single-function commit filter (where applicable) | CWE is attributed at the CVE/commit level; when multiple functions share a commit the per-function attribution is ambiguous |

### Datasets

| Dataset | Positives | Negatives | Source | Format |
|---------|-----------|-----------|--------|--------|
| **PrimeVul** | `target=1`, CWE non-empty, single-function commit | All `target=0` (explicit in dataset) | Hugging Face `benjis/primevul` | JSONL / Parquet |
| **ICVul** | `before_change=True`, `fc_hash` in CVE-FC mapping, CWE non-empty | None (VCC-only dataset) | Google Drive archive | CSV |
| **CVEfixes** | C/C++ language, CWE non-empty, single-function commit | None | Hugging Face `hitoshura25/cvefixes` | Parquet |
| **MegaVul** | CWE non-empty, single-function commit, optional CVSS threshold | None | Hugging Face `hitoshura25/megavul` | Parquet |
| **SecVulEval** | `is_vulnerable=True`, CWE parsed from inline `cwe_list` | All `is_vulnerable=False` (explicit) | Hugging Face `arag0rn/SecVulEval` | CSV |

**ICVul note:** The `function_info.csv` table uses `hash` = `fc_hash` (the fix commit), which maps to the `fc_hash` column of `cve_fc_vcc_mapping.csv`. The `vcc_hash` column in the same table is a separate set of hashes (vulnerability-contributing commits identified by SZZ) and is not used for the join.

**MegaVul CVSS threshold:** `extract_megavul(path, cvss_threshold=7.0)` drops samples with CVSS below the threshold or with null CVSS. Off by default.

### CWE classification

`cwe_label(cwe_str)` classifies each CWE ID against `cwec_latest.xml`:

| Label | Excel colour | Meaning |
|-------|-------------|---------|
| `leaf` | Green | Most specific weakness; no children in MITRE tree |
| `non-leaf` | Amber | Parent/intermediate node; has children (broad attribution) |
| `deprecated` | Pink | Superseded weakness; name starts with `DEPRECATED:` |
| `category` | Purple | MITRE organisational grouping (e.g. 7PK categories); not a weakness |
| `unknown` | Gray | ID not found in `cwec_latest.xml` |

---

## Notebooks

### `00_download_datasets.ipynb` -- Download

Downloads all five datasets to `data/raw/` and verifies the expected files are present. Each section is idempotent (skips if the file already exists).

Run this notebook **once** before anything else.

**Downloads:**
1. **PrimeVul** -- via Hugging Face `datasets` library; saves train and test splits as Parquet
2. **ICVul** -- via `gdown` from a shared Google Drive archive; extracts 5 CSV tables to `data/raw/icvul/`
3. **CVEfixes** -- via Hugging Face `hitoshura25/cvefixes`; 3 Parquet shards to `data/raw/cvefixes/`
4. **MegaVul** -- via Hugging Face `hitoshura25/megavul`; 2 Parquet shards to `data/raw/megavul/`
5. **SecVulEval** -- via Hugging Face `arag0rn/SecVulEval`; saved as `data/raw/secvuleval.csv`
6. Verify step: confirms all expected files are present before proceeding

### `01_datasets_composition.ipynb` -- Stage 1 analysis

Loads all five datasets, computes statistics, and writes the Stage 1 deliverable.

**Cells:**
1. Path setup and constants
2. Download `cwec_latest.xml` from MITRE (if not already present)
3. Load `CWENavigator` and define `cwe_label()`
4. Extract samples from each dataset (datasets not yet downloaded are skipped with a warning)
5. Compute per-dataset statistics (total, vulnerable, safe, unique CWEs, per-CWE counts)
6. Build and save `outputs/stage1_dataset_stats.xlsx`
7. Print CWE granularity breakdown and non-leaf detail table

---

## Deliverables

### `outputs/stage1_dataset_stats.xlsx`

One sheet per language group (currently `C_C++`), plus a `Legend` sheet.

**Sheet layout:**

| Dataset | Total Samples | Vulnerable | Safe | Unique CWEs | CWE-476 | CWE-416 | ... |
|---------|--------------|------------|------|-------------|---------|---------|-----|

- Fixed columns are dark-blue-header; data rows alternate white/light-gray.
- Each CWE column header is colour-coded by granularity (see table above).
- Header cells show both the CWE ID and its MITRE name (wrapped).
- Counts are blank (not zero) for datasets that have no sample under a given CWE.
- First column and header row are frozen for scrolling.

---

## Setup

```bash
pip install -e ".[dev]"
```

Requires Python >= 3.10.

### Run tests

```bash
pytest
```

41 unit tests covering all five extractors and the `FunctionSample` schema.

### Run notebooks

```bash
# download datasets first
jupyter notebook notebooks/00_download_datasets.ipynb

# then produce the Stage 1 report
jupyter notebook notebooks/01_datasets_composition.ipynb
```

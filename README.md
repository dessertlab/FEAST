# FEAST

**Fusing Evidence Across Static Analysis Tools for CWE-Specific Vulnerability Detection**

Multi-language vulnerability dataset pipeline. Collects, normalises, and synthesises labelled code samples from 24 public sources across C/C++, Java, and Python, each annotated with CWE IDs from the MITRE catalogue.

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

Stage 4  Tool execution            (planned)
Stage 5  Per-CWE metric computation (planned)
Stage 6  DST fusion                 (planned)
```

Stages 0–3 are fully implemented. Stages 4–6 are planned.

---

## Repository structure

```
FEAST/
├── main.py                          # CLI (see Usage below)
├── ingestion/                       # extraction library
│   ├── schema.py                    # FunctionSample dataclass
│   ├── cwe_navigator.py             # MITRE CWE XML parser and tree walker
│   ├── utils.py                     # shared helpers (CWE regex, NVD placeholders)
│   └── <source>.py                  # one extractor module per dataset
├── notebooks/
│   ├── 00_download_datasets.ipynb   # Stage 0 – download
│   ├── 01_c_cpp.ipynb               # Stage 1 – C/C++ statistics
│   ├── 01_java.ipynb                # Stage 1 – Java statistics
│   ├── 01_python.ipynb              # Stage 1 – Python statistics
│   ├── 02_synthesis.ipynb           # Stage 2 – process + merge
│   └── 03_materialize.ipynb         # Stage 3 – write source files
├── data/
│   ├── raw/                         # downloaded datasets (git-ignored)
│   ├── cwec_latest.xml              # MITRE CWE catalogue (auto-downloaded)
│   ├── processed/                   # per-dataset CWE-filtered parquets
│   │   ├── c_cpp/
│   │   ├── java/
│   │   └── python/
│   ├── merged/                      # final deduplicated parquets
│   │   ├── c_cpp_merged.parquet
│   │   ├── java_merged.parquet
│   │   └── python_merged.parquet
│   └── materialized/                # individual source files for static analysis
│       ├── c_cpp/
│       │   ├── <dataset>/           # one directory per source dataset
│       │   │   └── <id>.c           # one file per sample
│       │   └── index.parquet        # sample_id -> source, label, cwes, branch
│       ├── java/
│       └── python/
├── outputs/
│   ├── stage1_c_cpp_stats.xlsx
│   ├── stage1_java_stats.xlsx
│   └── stage1_python_stats.xlsx
├── tests/ingestion/                 # unit tests for all extractors
└── pyproject.toml
```

---

## Datasets

25 sources organised by language and branch.

### C/C++ — 11 sources

| Dataset | Branch | Positives | Negatives |
|---------|--------|-----------|-----------|
| PrimeVul | real | `target=1`, single-function commit, CWE non-empty | all `target=0` (explicit) |
| ICVul | real | `before_change=True`, `fc_hash` in CVE-FC mapping | none |
| CVEfixes(C) | real | C/C++ language, single-function commit, CWE non-empty | none |
| MegaVul | real | single-function commit, CWE non-empty | none |
| SecVulEval | real | `is_vulnerable=True` | all `is_vulnerable=False` |
| CrossVul(C) | real | `bad_*` files (vulnerable functions) | `good_*` files (fix-paired) |
| SVEN(C) | real | `func_src_before`, CWE from `vul_type` | `func_src_after` (fix-paired) |
| Juliet(C) | synth | `*_bad.c` files | `*_good*.c` files |
| CASTLE | synth | `vulnerable=True` | `vulnerable=False` |
| FormAI | ai | `VULNERABLE` / ESBMC `error_type` mapped to CWE | `NON-VULNERABLE` / safe samples |
| LLMSecEval(C) | ai | `gen_scenario/*.c` (Copilot completions) | none |

### Java — 5 sources

| Dataset | Branch | Positives | Negatives |
|---------|--------|-----------|-----------|
| CVEfixes(Java) | real | Java language, single-function commit | none |
| CrossVul(Java) | real | `bad_*` files | `good_*` files (fix-paired) |
| Juliet(Java) | synth | `*_bad.java` files | `*_good*.java` files |
| OWASP(Java) | synth | `real vulnerability=true` | `real vulnerability=false` |
| CAPEC_LLM(Java) | ai | LLM-generated snippets for CAPEC entries | none |

### Python — 9 sources

| Dataset | Branch | Positives | Negatives |
|---------|--------|-----------|-----------|
| CVEfixes(Python) | real | Python language, single-function commit | none |
| PatchEval | real | `vul_func` where `language=Python` | `fix_func` (fix-paired) |
| CrossVul(Python) | real | `bad_*` files | `good_*` files (fix-paired) |
| PyVul | real | `code_before`, CWE from commits map | `code_after` (fix-paired) |
| SVEN(Python) | real | `func_src_before` | `func_src_after` (fix-paired) |
| OWASP(Python) | synth | `real vulnerability=true` | `real vulnerability=false` |
| LLMSecEval | ai | `gen_scenario/*.py` (Copilot completions) | `Secure/*.py` files |
| SecurityEval | ai | all samples (vulnerable-only dataset) | none |
| CAPEC_LLM(Python) | ai | LLM-generated snippets for CAPEC entries | none |

**Branch semantics:**

| Branch | Meaning |
|--------|---------|
| `real` | Functions extracted from actual CVE patches or real-world codebases |
| `synth` | Template/rule-based synthesised code (Juliet test suite, OWASP Benchmark) |
| `ai` | LLM-generated code (Copilot completions, ChatGPT-generated snippets) |

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

| Column | Description |
|--------|-------------|
| `source` | Dataset name (e.g. `"PyVul"`) |
| `code_hash` | Full SHA-256 of normalised code (used for deduplication) |
| `sample_id` | First 16 hex chars of `code_hash`; used as filename stem in Stage 3 |

---

## CWE classification

The pipeline classifies every CWE ID against `cwec_latest.xml` from MITRE:

| Type | Meaning |
|------|---------|
| `leaf` | Most specific weakness; no children in MITRE hierarchy |
| `non-leaf` | Parent or intermediate node (broad attribution) |
| `category` | MITRE organisational grouping, not a proper weakness |
| `deprecated` | Superseded weakness (name starts with `DEPRECATED:`) |
| `unknown` | ID not found in `cwec_latest.xml` |

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

---

## CLI

`main.py` exposes the full pipeline from the command line.

### `download` — fetch all datasets (Stage 0)

```bash
uv run python main.py download
```

Downloads all 17 datasets to `data/raw/`. Idempotent: already-present paths are skipped. Two datasets require manual download from Zenodo; the command prints instructions for these when they are missing:

| Dataset | File to place in `data/raw/` |
|---------|------------------------------|
| CrossVul | `crossvul.zip` |
| LLMSecEval (vulnerable) | `copilot-cwe-scenarios-dataset.zip` (Zenodo record 5225651) |

### `list` — show all available sources

```bash
uv run python main.py list
```

### `synthesize` — build processed and merged parquets

```bash
uv run python main.py synthesize [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--lang LANG` | `all` | Language to process: `c`, `java`, `python`, or `all` |
| `--sources SRC1,SRC2,...` | all available | Comma-separated dataset names to include |
| `--cwe-types TYPES` | `leaf,non-leaf` | CWE node types to retain for vulnerable samples |
| `--cwes CWE-79,CWE-89,...` | all | Explicit whitelist of CWE IDs |
| `--min-cwe-count N` | `1` (off) | Drop CWEs with fewer than N vulnerable samples after merge |
| `--branches BRANCHES` | `all` | Source branches to include: `real`, `synth`, `ai`, or comma-separated |
| `--data-dir DIR` | `data/raw/` | Override the raw data directory |

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

| Option | Default | Description |
|--------|---------|-------------|
| `--lang LANG` | `all` | Language to materialize: `c`, `java`, `python`, or `all` |
| `--overwrite` | off | Re-write files that already exist |

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

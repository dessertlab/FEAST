# FEAST — Design of the analysis pipeline (canonical-family fusion)

This document records the methodology and module layout of the analysis half of FEAST
(Stage 5 calibration + fusion). The ingestion/synthesis/enrichment half is documented in
`README.md` and is unchanged.

## 1. Why canonical families

The earlier pipeline matched tool output and ground truth with an **asymmetric** rule:
the tool side required the *exact* CWE, the ground-truth side accepted any *vertically
related* CWE (ancestor/descendant closure). On a per-CWE one-vs-rest grid this caused:

* **TP+FN double scoring** — a tool firing CWE-564 on a CWE-89 row scored a TP for 564
  *and* an FN for 89, inflating false negatives (the metric the project targets);
* **no evidence pooling** — related CWEs were voted in isolation, so a tool calibrated on
  a parent CWE never contributed to a sibling/child during fusion, even though
  calibration had credited it for exactly those cases. Calibration and fusion disagreed.

**Fix:** collapse both tool outputs and ground truth to a **CWE family** — an ancestor in
the MITRE CWE-1000 (Research Concepts) hierarchy — and then match families **exactly**.
The hierarchy is consulted in exactly one place (canonicalisation); everything downstream
is plain set membership, identical in calibration and fusion.

## 2. Family definition (primary-path rule)

CWE-1000 is a DAG: ~22% of the CWEs in the dataset have ≥2 ChildOf parents. To get a
deterministic partition we follow only the parent marked `Ordinal="Primary"` within
`View_ID="1000"` (`CWENavigator.primary_parent` / `primary_path`). This disambiguates
**100%** of the multi-parent cases (538/545 dataset weaknesses have exactly one
primary-1000 parent; 7 are roots; 0 ambiguous) and discards misleading cross-view edges
(e.g. CWE-120's `CWE-20` parent, which belongs to view 700, not 1000).

Three levels are supported (`analysis/canonical.py`, `CweCanonicalizer`):

| level         | rule                                              | example mappings |
|---------------|---------------------------------------------------|------------------|
| `pillar`      | top weakness of the view | 79,89,78 → 707 ; 120,125 → 664 |
| `subcategory` | project-defined node directly under the pillar; not a MITRE `Abstraction` value | 79,89,78 → 74 ; 120,125 → 118 |
| `class`       | nearest ancestor with `Abstraction=Class`; if a branch has no Class node, fall back to the node directly under the pillar, preserving `Pillar -> Base` branches | 79 → 74, 89 → 943, 78 → 77 ; 120,125 → 119 ; 1024 → 1024 |

**`class` is the recommended default.** Measured on the Python grid, `pillar` and
`subcategory` over-merge the highest-traffic family (all injection collapses into CWE-74),
destroying the SQLi/XSS/command-injection distinctions where tools specialise. `class`
keeps those distinct *and* merges trivial leaf splits (buffer over-read/over-write → 119),
because MITRE itself draws a `Class` boundary at SQLi (943) but not between buffer
variants. The level is an explicit experiment axis (`--level pillar|subcategory|class|all`)
so the over-merging effect can be reported as an ablation.

## 3. Matching, confusion, calibration

After canonicalisation a tool "fires" family `f` iff `f` is in its family set, and the GT
is positive for `f` iff `f` is in the GT family set — **exact, symmetric** membership.
`analysis/calibration.py` builds the one-vs-rest confusion per `(tool, family)`
(vectorised with boolean incidence matrices) and derives `ppv` (fire credibility), `npv`
(silence credibility), `fpr`/`fnr`, their complements, `positive_support` and `supported`
(`tp+fp>0`). These reliability numbers are the fusion configuration.

## 4. Fusion strategies (`analysis/fusion/`)

One package, one strategy per module, sharing the plumbing in `common.py` (fire index,
exact labels, metric lookup, prediction schema):

* `weighted.py` — reliability-weighted voting (4 fire/silence metric pairs);
* `traditional.py` — K-of-N baseline (all tools and supported-only);
* `dst.py` — Dempster-Shafer (Dempster, PCR6, Yager) on the binary frame {V,S}, expanded over the same 4 fire/silence metric pairs;
* `bayes.py` — naive Bayes over per-tool log-likelihood ratios;
* `bks.py` — empirical behavior-knowledge-space lookup over tool fire patterns;
* `logistic.py` — per-family logistic regression on the fire indicators.

Noisy-OR is intentionally not part of the supported strategy set. Every scored fusion
strategy is materialised at the discrete thresholds `τ ∈ {0.1, …, 0.9}` as explicit
strategy names (for example `naive_bayes_tau_0_7`), so per-family, detection, plots and
CSV reports all operate on the same strategy identifiers. Baseline single-tool, OR, and
traditional K-of-N rows remain unswept discrete references. Every strategy emits the same
per-(row, family) schema, so they all feed `predictions.evaluate_predictions` (metrics per
`(strategy, family)`) and `detection_from_predictions` (per-row vuln/safe collapse +
`cwe_attribution_acc`).

## 5. Two-stage aggregation (`analysis/aggregation.py`)

1. **mean over folds** — average each metric across the K folds per `(strategy, family)`;
   family support is *summed* (→ total GT occurrences).
2. **support-weighted mean over families** — per `(strategy, metric)`, weight the family
   values by that support. This weighted value is the headline; the macro mean and median
   are kept as context.

## 6. Restriction (`analysis/experiment.py`)

A family is analysed iff it is **supported** (fired by ≥1 tool) **and** has **≥K
ground-truth occurrences** (`K = --min-cwe-count`, default `= --n-splits`; the necessary
condition for a per-fold-stable confusion matrix). The header prints how many families are
kept vs present, why the rest dropped, and the fraction of GT occurrences retained — e.g.
Python/`class`: *35 kept / 112 present (74 unsupported, 3 below K); 71.2% of GT
occurrences covered*.

## 7. Folding

`analysis/folds.py` — multilabel-stratified k-fold (`iterative-stratification`, a hard
dependency for reproducibility) on the **canonical family** labels, with rare-family
pruning at the same K. `before`/`after` fix-paired rows are split per-row (documented as
*correlated observations*, not leakage, because fusion features are tool verdicts, not
code).

## 8. Module map & outputs

```
ingestion/cwe_navigator.py   primary_parent / primary_path / abstraction (View_ID+Ordinal)
analysis/canonical.py        CWE -> family (pillar/subcategory/class)
analysis/folds.py            stratified k-fold + rare-family pruning
analysis/calibration.py      exact per-(tool, family) confusion + reliability
analysis/fusion/             strategies + predictions + detection
analysis/aggregation.py      two-stage aggregation
analysis/reporting.py        CSV + SVG/PNG writers
analysis/experiment.py       CV orchestration per (language, level)
main.py  fusion --level …    thin CLI entry point
```

Outputs per `data/results/<language>/<level>/`: `config.json` (run config + restriction
stats), `canonical_map.csv`, `folds.csv`, `calibration_reliability.csv`,
`fusion_metrics_per_family.csv`, `fusion_metrics_overall.csv`,
`fusion_detection_overall.csv`, `fusion_tau_sweep.csv`, `fusion_operating_points.csv`, and
`plots/<metric>.{svg,png}`. The `fusion_tau_sweep.csv` and operating-point reports are
derived from the explicit `*_tau_*` strategy rows rather than from a hidden post-hoc
thresholding pass.

## 9. Key decisions

* Matching is **exact on families**; the old vertical-closure matcher is removed.
* Support weight = `positive_support` (GT occurrences), summed across folds.
* All three levels run by default; `class` is the recommended headline.
* Restriction = supported ∧ ≥K GT occurrences.
* Superseded modules (`metrics`, `split`, `report`, `visualization`, monolithic `fusion`)
  were removed rather than deprecated; the strategy math was preserved and relocated.

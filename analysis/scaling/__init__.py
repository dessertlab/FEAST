"""Scaling / ablation studies: *why* fusion performance differs across languages.

Each module is one lens on the same question — does the per-language gap track the number
of tools, the dataset size, or the tool-pool coverage, rather than the language itself?

* ``coverage_xlang``  — cross-language oracle/union-recall table (reuses ``complementarity``).
* ``tool_ablation``   — vary the number of tools *within* a language (breaks the
  language/n-tools collinearity that N=3 languages alone cannot).
* ``data_ablation``   — vary the dataset size within a language (learning curve).
* ``meta_regression`` — stack every (family, fold, strategy, language) observation and fit a
  mixed-effects model that separates the dataset-level drivers from the language label.

All lenses operate on the canonical family space from ``analysis.experiment.prepare_canonical``
and write under ``data/results/`` next to the fusion outputs they explain.
"""

from analysis.scaling.common import (
    LANGUAGES,
    dataset_covariates,
    fold_weighted_f1,
    paired_delta_per_family_fold,
)
from analysis.scaling.coverage_xlang import compare_coverage
from analysis.scaling.data_ablation import ablate_dataset_size
from analysis.scaling.meta_regression import build_stack, run_meta
from analysis.scaling.tool_ablation import ablate_tools

__all__ = [
    "LANGUAGES",
    "dataset_covariates",
    "fold_weighted_f1",
    "paired_delta_per_family_fold",
    "compare_coverage",
    "ablate_tools",
    "ablate_dataset_size",
    "build_stack",
    "run_meta",
]

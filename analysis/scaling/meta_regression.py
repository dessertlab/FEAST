"""Analysis 4 — the model that puts every lens together.

Goal: explain the paired fusion lift (``delta_f1`` over the traditional baseline) as a
function of the *dataset-level* drivers — number of tools, number of findings, tool-pool
coverage — rather than the bare language label, so the per-language gap is attributed to a
mechanism that would generalise to a fourth language.

Two complementary fits, because they answer the identification problem differently:

* **mixed-effects model** on the stacked per-(family, fold) observations of the three
  canonical runs, with a random intercept on ``family`` (and ``language``). This is the
  honest cross-language association, but with only three languages the language-level
  covariates are weakly identified (they are nearly collinear) — read it together with:

* **within-language ablation regressions** on the ``by_n_tools`` / ``by_dataset_size`` tables
  produced by analyses 2 and 3. Holding the language fixed (via language dummies) these
  identify the tool-count and dataset-size slopes from variation that exists *inside* a
  language, which is exactly what the three-point cross-language comparison cannot do.

The mixed model needs ``statsmodels`` (declared in pyproject; run ``uv sync``). The ablation
regressions use a dependency-free OLS with cluster-robust standard errors, so they run with
only numpy/pandas present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.aggregation import SUPPORT_COLUMN
from analysis.scaling.common import (
    LANGUAGES,
    dataset_covariates,
    load_per_family_per_fold,
    paired_delta_per_family_fold,
)
from analysis.scaling.coverage_xlang import CROSS_LANG_DIRNAME

META_DIRNAME = "meta_regression"


# ── data assembly ──────────────────────────────────────────────────────────────

def build_stack(
    languages: list[str] | None = None,
    *,
    tier: str = "full",
    metric: str = "f1",
    baseline: str = "traditional",
    results_root: str | Path = "data/results",
) -> pd.DataFrame:
    """Stack per-(family, fold) fusion lift across languages with their covariates.

    For each language the per-family-per-fold paired ``delta`` is averaged over fusion
    strategies (one robust outcome per family/fold, no post-hoc strategy cherry-picking),
    then joined with the language-level covariates and, when available, the union recall from
    the cross-language coverage table. ``family`` is kept as the grouping key for the random
    intercept.
    """
    languages = languages or LANGUAGES
    coverage = _load_coverage(results_root)

    frames = []
    for language in languages:
        try:
            pf = load_per_family_per_fold(language, tier, results_root=results_root)
            cov = dataset_covariates(language, tier, results_root=results_root)
        except FileNotFoundError as exc:
            print(f"[skip] {language}: {exc}")
            continue
        delta = paired_delta_per_family_fold(pf, metric=metric, baseline=baseline)
        if delta.empty:
            continue
        # collapse strategies: mean lift per (family, fold), support taken from any row
        grouped = (
            delta.groupby(["family", "fold"], as_index=False)
            .agg(delta=("delta", "mean"), family_support=(SUPPORT_COLUMN, "first"))
        )
        grouped["language"] = language
        grouped["n_tools"] = cov["n_tools"]
        grouped["log_gt_kept"] = np.log10(cov["gt_kept"]) if cov["gt_kept"] else np.nan
        grouped["log_n_samples"] = np.log10(cov["n_samples"]) if cov["n_samples"] > 0 else np.nan
        grouped["kept_frac"] = cov["kept_frac"]
        grouped["union_recall"] = coverage.get(language, np.nan)
        grouped["log_family_support"] = np.log10(grouped["family_support"].clip(lower=1))
        frames.append(grouped)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_coverage(results_root: str | Path) -> dict[str, float]:
    path = Path(results_root) / CROSS_LANG_DIRNAME / "coverage.csv"
    if not path.exists():
        return {}
    cov = pd.read_csv(path)
    return dict(zip(cov["language"], cov["family_or_recall"]))


# ── dependency-free OLS with cluster-robust SE ──────────────────────────────────

def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray, names: list[str]) -> pd.DataFrame:
    """OLS with cluster-robust (sandwich) standard errors clustered on ``groups``.

    ``X`` must already include the intercept column. Returns a coefficient table with
    cluster-robust SE, t and two-sided p (normal approximation). Used for the within-language
    ablation regressions, where the clusters are languages (or families) and no random
    effect is needed.
    """
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta

    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        Xg = X[groups == g]
        ug = resid[groups == g]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    cov = XtX_inv @ meat @ XtX_inv

    n_clusters = len(np.unique(groups))
    adj = n_clusters / max(n_clusters - 1, 1)
    se = np.sqrt(np.diag(cov) * adj)
    from statistics import NormalDist
    t = beta / se
    p = np.array([2 * (1 - NormalDist().cdf(abs(ti))) for ti in t])
    return pd.DataFrame({"term": names, "coef": beta, "cluster_se": se, "t": t, "p_value": p})


def _design_with_dummies(df: pd.DataFrame, numeric: list[str], factor: str) -> tuple[np.ndarray, list[str]]:
    """Intercept + numeric columns + one-hot dummies of ``factor`` (first level dropped)."""
    cols = [np.ones(len(df))]
    names = ["intercept"]
    for col in numeric:
        cols.append(df[col].to_numpy(dtype=float))
        names.append(col)
    levels = sorted(df[factor].astype(str).unique())[1:]  # drop reference level
    for lvl in levels:
        cols.append((df[factor].astype(str) == lvl).to_numpy(dtype=float))
        names.append(f"{factor}={lvl}")
    return np.column_stack(cols), names


def fit_ablation_regression(table: pd.DataFrame, predictor: str, outcome: str = "fusion_f1_max") -> pd.DataFrame:
    """Within-language slope of ``outcome`` on ``predictor`` with language fixed effects.

    Clusters SE on language. Returns the coefficient table; the ``predictor`` row is the
    identified within-language effect (e.g. the tool-count or log-dataset-size slope).
    """
    df = table.dropna(subset=[predictor, outcome]).copy()
    if df.empty or df["language"].nunique() == 0:
        return pd.DataFrame()
    X, names = _design_with_dummies(df, [predictor], "language")
    y = df[outcome].to_numpy(dtype=float)
    groups = df["language"].astype(str).to_numpy()
    return ols_cluster(y, X, groups, names)


# ── statsmodels mixed-effects model ──────────────────────────────────────────────

def drop_collinear(df_X: pd.DataFrame, tolerance: float = 1e-9) -> list[str]:
    cols = list(df_X.columns)
    if "Intercept" in cols:
        cols.remove("Intercept")
        cols = ["Intercept"] + cols
    
    kept = []
    for col in cols:
        if not kept:
            kept.append(col)
            continue
        X_sub = df_X[kept].to_numpy()
        col_vec = df_X[col].to_numpy()
        pinv = np.linalg.pinv(X_sub)
        proj = X_sub @ (pinv @ col_vec)
        resid = col_vec - proj
        norm_resid = np.linalg.norm(resid)
        norm_col = np.linalg.norm(col_vec)
        rel_resid = norm_resid / norm_col if norm_col > 1e-9 else norm_resid
        if rel_resid > tolerance:
            kept.append(col)
    return kept


def fit_mixed(stack: pd.DataFrame, *, group: str = "family") -> object:
    """Mixed model: delta ~ dataset covariates, random intercept on ``group``.

    Requires statsmodels. Returns the fitted results object (caller saves its summary).
    """
    try:
        import statsmodels.formula.api as smf
        import patsy
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "fit_mixed needs statsmodels and patsy — add them and run `uv sync`."
        ) from exc

    predictors = ["n_tools", "log_gt_kept", "kept_frac", "log_family_support", "union_recall"]
    usable = [p for p in predictors if stack[p].notna().any() and stack[p].nunique() > 1]
    data = stack.dropna(subset=usable + ["delta", group]).reset_index(drop=True)

    # Use patsy to build the full fixed-effects design matrix to identify collinearity
    full_formula = "delta ~ " + " + ".join(usable)
    _, exog = patsy.dmatrices(full_formula, data, return_type="dataframe")

    # Drop collinear columns, preserving Intercept if present
    independent_cols = drop_collinear(exog)
    independent_predictors = [c for c in independent_cols if c != "Intercept"]

    formula = "delta ~ " + " + ".join(independent_predictors)

    dropped = [p for p in usable if p not in independent_predictors]
    if dropped:
        print(f"[warn] Dropped collinear predictors in mixed model: {dropped}")

    model = smf.mixedlm(formula, data=data, groups=data[group])
    return model.fit(reml=True)


# ── orchestration ────────────────────────────────────────────────────────────────

def run_meta(
    languages: list[str] | None = None,
    *,
    tier: str = "full",
    results_root: str | Path = "data/results",
) -> dict:
    """Assemble the stack, fit every model that its inputs allow, and write the report."""
    out_dir = Path(results_root) / CROSS_LANG_DIRNAME / META_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    stack = build_stack(languages, tier=tier, results_root=results_root)
    if stack.empty:
        print("[skip] meta-regression: no per-fold data found (run `python main.py fusion` first)")
        return {}
    stack.to_csv(out_dir / "stack.csv", index=False)
    report.append(f"stacked observations: {len(stack)} over {stack['language'].nunique()} languages, "
                  f"{stack['family'].nunique()} families")

    # within-language ablation regressions (run if the ablation tables exist)
    tools_tbl = _gather_ablation(results_root, languages, tier, "by_n_tools.csv")
    if not tools_tbl.empty:
        coef = fit_ablation_regression(tools_tbl, "n_tools")
        coef.to_csv(out_dir / "coef_tools.csv", index=False)
        report.append(_format_coef("Tool-count effect (within-language, cluster-robust)", coef, "n_tools"))

    data_tbl = _gather_ablation(results_root, languages, tier, "by_dataset_size.csv")
    if not data_tbl.empty:
        data_tbl["log_n_samples"] = np.log10(data_tbl["n_samples"].clip(lower=1))
        coef = fit_ablation_regression(data_tbl, "log_n_samples")
        coef.to_csv(out_dir / "coef_data.csv", index=False)
        report.append(_format_coef("Dataset-size effect (within-language, cluster-robust)", coef, "log_n_samples"))

    # cross-language mixed model (needs statsmodels)
    try:
        result = fit_mixed(stack)
        (out_dir / "mixed_model_summary.txt").write_text(str(result.summary()), encoding="utf-8")
        report.append("Mixed model fitted — see mixed_model_summary.txt")
        report.append(str(result.summary()))
    except ModuleNotFoundError as exc:
        report.append(f"Mixed model skipped: {exc}")

    summary_text = "\n\n".join(report)
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)
    print(f"\n[write] {out_dir}")
    return {"stack_rows": len(stack), "out_dir": str(out_dir)}


def _gather_ablation(results_root, languages, tier, filename) -> pd.DataFrame:
    languages = languages or LANGUAGES
    frames = []
    for lang in languages:
        path = Path(results_root) / lang / "pillar_child" / tier / "ablation" / filename
        if path.exists():
            frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _format_coef(title: str, coef: pd.DataFrame, key: str) -> str:
    if coef.empty:
        return f"{title}: (insufficient data)"
    row = coef[coef["term"] == key]
    body = coef.to_string(index=False)
    if not row.empty:
        r = row.iloc[0]
        head = f"{title}\n  {key}: coef={r['coef']:.4f}, cluster-SE={r['cluster_se']:.4f}, p={r['p_value']:.3g}"
    else:
        head = title
    return f"{head}\n{body}"

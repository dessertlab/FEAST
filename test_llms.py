"""
test_llms.py  —  few-shot LLM vulnerability detection benchmark
================================================================
Workflow:
  1. Load data/merged/python_merged.parquet
  2. Subsample 50 items: 20 safe + 10 per each of 3 target CWEs
  3. For every (sample, model) pair run N_RUNS independent completions
  4. Dump raw results to CSV
  5. Print accuracy / F1 per model and self-consistency per (model, sample)

Providers supported: Ollama (OpenAI-compatible) and NVIDIA Build (OpenAI-compatible).
Fill the placeholders in the CONFIG section before running.

CLI usage:
  python test_llms.py --provider ollama
  python test_llms.py --provider nvidia
  python test_llms.py --provider ollama,nvidia   # ollama primary, nvidia fallback
  python test_llms.py --provider nvidia,ollama   # nvidia primary, ollama fallback
"""

from __future__ import annotations

import argparse
import csv
import random
import textwrap
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from openai import OpenAI
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

_console = Console()

# ── CONFIG — fill these before running ────────────────────────────────────────

SEED = 42
N_RUNS = 3          # independent completions per (sample, model)
N_SAFE = 20         # safe samples to draw
N_PER_CWE = 10      # vulnerable samples per target CWE

TARGET_CWES: list[str] = [
    "CWE-79",   # XSS
    "CWE-89",   # SQL injection
    "CWE-22",   # path traversal
]

DATASET_PATH = Path("data/merged/python_merged.parquet")
OUTPUT_CSV   = Path("test_llms_results.csv")

# --- Ollama (local or cloud, OpenAI-compatible) --------------------------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"   # change for cloud endpoint
OLLAMA_API_KEY  = "5e154833fa2d46dd995a1ca210adeb49.UIekHkIAxaOnZN4nr1xJ2ilP"
OLLAMA_MODELS: list[str] = [
    "gpt-oss:120b-cloud",
    #"qwen3.5:cloud"
    "kimi-k2.6:cloud",
    #"glm-5.1:cloud"
]

# --- NVIDIA Build (OpenAI-compatible) ------------------------------------------
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY  = "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"
NVIDIA_MODELS: list[str] = [
    "moonshotai/kimi-k2.6",
    #"z-ai/glm-5.1",
    "qwen/qwen3.5-397b-a17b",
    "openai/gpt-oss-120b"
]

# Maximum tokens to feed from each code snippet (avoid context overflow)
MAX_CODE_CHARS = 3000

# ── VULNERABILITY DETECTION PROMPT ────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a security code auditor specialising in Python.
    Analyse the provided Python function and respond with exactly one line:
        LABEL: <0,None | 1,CWE-XXX>

    LABEL format rules:
      - Safe function  →  LABEL: 0,None
      - Vulnerable     →  LABEL: 1,CWE-XXX   (replace XXX with the CWE number)

    Do not include anything else in your response.
""")

# Few-shot examples shown as user/assistant pairs before the real query.
# Deliberately simple and not from the dataset.
_FEW_SHOT: list[dict] = [
    # --- safe example ---
    {
        "role": "user",
        "content": textwrap.dedent("""\
            Analyse the following Python function for security vulnerabilities:

            ```python
            import hashlib

            def hash_password(password: str) -> str:
                salt = os.urandom(16)
                dk = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 200_000)
                return salt.hex() + ':' + dk.hex()
            ```
        """),
    },
    {
        "role": "assistant",
        "content": "LABEL: 0,None",
    },
    # --- vulnerable example (CWE-89 SQL injection) ---
    {
        "role": "user",
        "content": textwrap.dedent("""\
            Analyse the following Python function for security vulnerabilities:

            ```python
            def get_user(db, username):
                query = "SELECT * FROM users WHERE username = '" + username + "'"
                return db.execute(query).fetchone()
            ```
        """),
    },
    {
        "role": "assistant",
        "content": "LABEL: 1,CWE-89",
    },
]

USER_TEMPLATE = textwrap.dedent("""\
    Analyse the following Python function for security vulnerabilities:

    ```python
    {code}
    ```
""")

# ── PROVIDER SETUP ─────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    name: str                       # display name, e.g. "ollama/qwen2.5-coder:7b"
    model_id: str                   # model string sent to the API
    client: OpenAI
    fallback: "ModelConfig | None" = None  # tried if primary raises an exception


def _make_ollama_configs(fallback: "ModelConfig | None" = None) -> list[ModelConfig]:
    client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
    return [
        ModelConfig(name=f"ollama/{mid}", model_id=mid, client=client, fallback=fallback)
        for mid in OLLAMA_MODELS
    ]


def _make_nvidia_configs(fallback: "ModelConfig | None" = None) -> list[ModelConfig]:
    client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
    return [
        ModelConfig(name=f"nvidia/{mid}", model_id=mid, client=client, fallback=fallback)
        for mid in NVIDIA_MODELS
    ]


def build_clients(provider_order: list[str]) -> list[ModelConfig]:
    """
    provider_order examples:
      ["ollama"]          → only Ollama models
      ["nvidia"]          → only NVIDIA models
      ["ollama", "nvidia"] → Ollama primary, NVIDIA fallback (first NVIDIA model)
      ["nvidia", "ollama"] → NVIDIA primary, Ollama fallback (first Ollama model)
    """
    if len(provider_order) == 1:
        p = provider_order[0]
        return _make_ollama_configs() if p == "ollama" else _make_nvidia_configs()

    primary_name, fallback_name = provider_order[0], provider_order[1]

    # Build a single fallback ModelConfig (first model of the fallback provider).
    if fallback_name == "nvidia":
        fb_client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY)
        fb = ModelConfig(
            name=f"nvidia/{NVIDIA_MODELS[0]}",
            model_id=NVIDIA_MODELS[0],
            client=fb_client,
        )
    else:
        fb_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
        fb = ModelConfig(
            name=f"ollama/{OLLAMA_MODELS[0]}",
            model_id=OLLAMA_MODELS[0],
            client=fb_client,
        )

    if primary_name == "ollama":
        return _make_ollama_configs(fallback=fb)
    else:
        return _make_nvidia_configs(fallback=fb)


# ── SAMPLING ──────────────────────────────────────────────────────────────────

def sample_dataset(path: Path, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)

    df = pd.read_parquet(path)
    df["cwes"] = df["cwes"].apply(
        lambda v: list(v) if isinstance(v, (list, np.ndarray)) else []
    )

    # safe samples
    safe_pool = df[df["label"] == 0].copy()
    safe_idx  = rng.sample(list(safe_pool.index), min(N_SAFE, len(safe_pool)))
    safe_df   = safe_pool.loc[safe_idx].copy()
    safe_df["target_cwe"] = "SAFE"

    # vulnerable samples — one group per target CWE
    vuln_frames: list[pd.DataFrame] = []
    for cwe in TARGET_CWES:
        pool = df[df["cwes"].apply(lambda cws: cwe in cws)].copy()
        if len(pool) < N_PER_CWE:
            print(f"[warn] {cwe}: only {len(pool)} samples available, using all")
        idx = rng.sample(list(pool.index), min(N_PER_CWE, len(pool)))
        chunk = pool.loc[idx].copy()
        chunk["target_cwe"] = cwe
        vuln_frames.append(chunk)

    result = pd.concat([safe_df] + vuln_frames, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle

    print(f"[info] subsample: {len(result)} rows  "
          f"(safe={len(safe_df)}, "
          + ", ".join(f"{cwe}={len(vf)}" for cwe, vf in zip(TARGET_CWES, vuln_frames))
          + ")")
    return result


# ── INFERENCE ─────────────────────────────────────────────────────────────────

import re as _re
_LABEL_RE = _re.compile(r"LABEL:\s*([01])\s*,\s*(CWE-\d+|None)", _re.IGNORECASE)


def parse_response(response_text: str) -> tuple[int | None, str | None]:
    """
    Returns (predicted_label, predicted_cwe).
    predicted_label: 0 or 1, or None on parse failure
    predicted_cwe:   "CWE-XX" string, "None" (literal), or None on parse failure
    """
    m = _LABEL_RE.search(response_text)
    if not m:
        return None, None
    label = int(m.group(1))
    cwe   = m.group(2).upper()          # "NONE" or "CWE-XX"
    return label, (None if cwe == "NONE" else cwe)


def _call_api(mc: ModelConfig, messages: list[dict]) -> str:
    response = mc.client.chat.completions.create(
        model=mc.model_id,
        messages=messages,
        temperature=0.0,
        max_tokens=1024,  # thinking models need room to complete CoT before content
    )
    return response.choices[0].message.content or ""


def run_single(mc: ModelConfig, code: str) -> tuple[str, int | None, str | None, str]:
    """
    Returns (raw_text, predicted_label, predicted_cwe, used_provider).
    Falls back to mc.fallback on API exception.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_FEW_SHOT,
        {"role": "user", "content": USER_TEMPLATE.format(code=code[:MAX_CODE_CHARS])},
    ]
    try:
        raw = _call_api(mc, messages)
        used = mc.name
    except Exception as exc:
        if mc.fallback is None:
            raise
        print(f"  [fallback] {mc.name} failed ({exc}), retrying with {mc.fallback.name}")
        raw  = _call_api(mc.fallback, messages)
        used = mc.fallback.name

    label, cwe = parse_response(raw)
    return raw, label, cwe, used


# ── ROTATION RUNNER ───────────────────────────────────────────────────────────

_RETRY_BASE  = 5    # seconds for first 429 retry
_RETRY_CAP   = 120  # max seconds between retries


def _run_sample_rotated(
    models: list[ModelConfig],
    code: str,
    sid: str,
    gt_label: int,
    gt_cwe: str,
    sample_num: int,
    total_samples: int,
    global_done: list[int],   # mutable counter [completed_runs]
    total_runs: int,
) -> list[dict]:
    """
    Run all (model × N_RUNS) combinations for one sample using a rotating queue.
    On 429, the item is re-queued with an exponential cooldown; other models
    continue in the meantime. Sleeps only when every pending item is throttled.
    Non-429 errors are recorded as-is (no retry).
    """
    # Each item: (mc, run_idx, attempts, not_before_timestamp)
    pending: deque = deque(
        (mc, run_idx, 0, 0.0)
        for mc in models
        for run_idx in range(N_RUNS)
    )
    results: list[dict] = []

    while pending:
        mc, run_idx, attempts, not_before = pending.popleft()

        # If this item is still in cooldown, put it back and maybe sleep
        now = time.monotonic()
        if now < not_before:
            pending.append((mc, run_idx, attempts, not_before))
            earliest = min(item[3] for item in pending)
            gap = earliest - time.monotonic()
            if gap > 0:
                print(f"  [throttled] all models in cooldown, sleeping {gap:.0f}s …")
                time.sleep(gap)
            continue

        global_done[0] += 1
        attempt_tag = f" attempt {attempts + 1}" if attempts > 0 else ""
        print(
            f"[{global_done[0]}/{total_runs}] "
            f"sample {sample_num}/{total_samples}  "
            f"model={mc.name}  run={run_idx + 1}/{N_RUNS}"
            f"{attempt_tag}"
        )

        try:
            raw, pred_label, pred_cwe, used_provider = run_single(mc, code)
            error = ""
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str:
                wait = min(_RETRY_BASE * (2 ** attempts), _RETRY_CAP)
                print(f"  [429] rate limited — requeueing {mc.name} run={run_idx + 1}, retry in {wait}s")
                pending.append((mc, run_idx, attempts + 1, time.monotonic() + wait))
                global_done[0] -= 1  # this attempt doesn't count as a completed run
                continue
            # non-retryable: record error and move on
            raw, pred_label, pred_cwe, used_provider, error = "", None, None, mc.name, err_str
            print(f"  [error] {err_str}")

        results.append({
            "sample_id":     sid,
            "gt_label":      gt_label,
            "gt_cwe":        gt_cwe,
            "model":         mc.name,
            "used_provider": used_provider,
            "run":           run_idx,
            "raw":           raw,
            "pred_label":    pred_label,
            "pred_cwe":      pred_cwe,
            "error":         error,
        })

    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM vulnerability detection benchmark")
    parser.add_argument(
        "--provider",
        default="ollama",
        help=(
            "Which provider(s) to use. Examples: "
            "ollama | nvidia | ollama,nvidia | nvidia,ollama. "
            "When two are given the second acts as fallback on API errors."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    provider_order = [p.strip().lower() for p in args.provider.split(",")]
    for p in provider_order:
        if p not in ("ollama", "nvidia"):
            raise SystemExit(f"Unknown provider '{p}'. Use 'ollama' and/or 'nvidia'.")

    random.seed(SEED)
    np.random.seed(SEED)

    dataset = sample_dataset(DATASET_PATH, SEED)
    models  = build_clients(provider_order)

    model_names = [m.name for m in models]
    print(f"[info] providers: {' → '.join(provider_order)}  |  models: {model_names}")

    rows: list[dict] = []
    total_samples = len(dataset)
    total_runs    = total_samples * len(models) * N_RUNS
    global_done   = [0]   # mutable so _run_sample_rotated can update it

    _console.rule("[bold]PER-SAMPLE DETAIL[/bold]")

    for sample_num, (_, sample) in enumerate(dataset.iterrows(), start=1):
        sid      = sample["sample_id"]
        code     = sample["code"]
        gt_label = int(sample["label"])
        gt_cwe   = sample["target_cwe"]

        sample_rows = _run_sample_rotated(
            models, code, sid, gt_label, gt_cwe,
            sample_num, total_samples, global_done, total_runs,
        )
        rows.extend(sample_rows)

        _print_one_sample(sample_num, sid, gt_label, gt_cwe, code, sample_rows, model_names)

    _console.rule(style="bright_black")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_CSV, index=False, quoting=csv.QUOTE_ALL)
    print(f"[info] raw results saved to {OUTPUT_CSV}")

    _print_report(results_df)


# ── PER-SAMPLE DETAIL PRINTER ─────────────────────────────────────────────────

def _print_one_sample(
    sample_num: int,
    sid: str,
    gt_label: int,
    gt_cwe: str,
    code: str,
    sample_rows: list[dict],
    model_names: list[str],
) -> None:
    # Header rule
    if gt_label == 1:
        gt_text = Text(f"VULNERABLE · {gt_cwe}", style="bold red")
    else:
        gt_text = Text("SAFE", style="bold green")

    title = Text()
    title.append(f"#{sample_num:02d}  ", style="bold")
    title.append("id: ", style="dim")
    title.append(sid[:20], style="cyan")
    title.append("  GT: ")
    title.append_text(gt_text)
    _console.print()
    _console.rule(title)

    # Code block (truncated to 25 lines)
    code_lines = code.splitlines()
    truncated  = "\n".join(code_lines[:25])
    if len(code_lines) > 25:
        truncated += f"\n… ({len(code_lines) - 25} more lines)"
    _console.print(Panel(
        Syntax(truncated, "python", theme="monokai", line_numbers=False),
        title="[dim]code[/dim]",
        border_style="bright_black",
        padding=(0, 1),
    ))

    # Per-model panels
    for model_name in model_names:
        model_rows = sorted(
            [r for r in sample_rows if r["model"] == model_name],
            key=lambda r: r["run"],
        )
        content = Text()
        preds: list[int] = []

        for rrow in model_rows:
            run_num    = int(rrow["run"]) + 1
            pred_label = rrow["pred_label"]
            pred_cwe   = rrow["pred_cwe"]
            used       = rrow["used_provider"]

            content.append(f"run {run_num}  LABEL: ", style="dim")

            if pred_label is None:
                content.append("PARSE ERROR", style="yellow")
                content.append("  ?", style="yellow")
            elif pred_label == gt_label:
                content.append(f"{pred_label},{pred_cwe or 'None'}", style="green")
                content.append("  ✓", style="bold green")
                preds.append(pred_label)
            else:
                content.append(f"{pred_label},{pred_cwe or 'None'}", style="red")
                content.append("  ✗", style="bold red")
                preds.append(pred_label)

            if used != model_name:
                content.append(f"  [fb:{used}]", style="yellow")

            content.append("\n")

        if len(preds) == N_RUNS:
            content.append("\n")
            if len(set(preds)) == 1:
                content.append(f"● consistent ({N_RUNS}/{N_RUNS} agree)", style="bold green")
            else:
                content.append(
                    f"⚠ inconsistent — {sum(preds)}/{N_RUNS} said VULNERABLE",
                    style="bold yellow",
                )

        _console.print(Panel(
            content,
            title=f"[dim]{model_name}[/dim]",
            border_style="bright_black",
            padding=(0, 1),
        ))


# ── METRICS ───────────────────────────────────────────────────────────────────

def _majority_label(series: pd.Series) -> int:
    return int(series.mean() >= 0.5)


def _print_report(df: pd.DataFrame) -> None:
    valid = df[df["pred_label"].notna()].copy()
    valid["pred_label"] = valid["pred_label"].astype(int)

    # ── 1. Global performance ─────────────────────────────────────────────────
    _console.rule("[bold]PERFORMANCE vs GROUND TRUTH[/bold]  (majority vote across runs)")
    t = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t.add_column("Model")
    t.add_column("Acc", justify="right")
    t.add_column("Prec", justify="right")
    t.add_column("Rec", justify="right")
    t.add_column("F1", justify="right")

    for model_name, mdf in valid.groupby("model"):
        ps = (
            mdf.groupby("sample_id")
            .agg(gt_label=("gt_label", "first"), pred_label=("pred_label", _majority_label))
            .reset_index()
        )
        y_true, y_pred = ps["gt_label"].tolist(), ps["pred_label"].tolist()
        t.add_row(
            model_name,
            f"{accuracy_score(y_true, y_pred):.3f}",
            f"{precision_score(y_true, y_pred, zero_division=0):.3f}",
            f"{recall_score(y_true, y_pred, zero_division=0):.3f}",
            f"{f1_score(y_true, y_pred, zero_division=0):.3f}",
        )
    _console.print(t)

    # ── 2. Per-CWE performance ────────────────────────────────────────────────
    _console.rule("[bold]PERFORMANCE per CWE[/bold]")
    t2 = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t2.add_column("Model")
    t2.add_column("CWE")
    t2.add_column("Acc", justify="right")
    t2.add_column("F1",  justify="right")
    t2.add_column("n",   justify="right")

    for model_name, mdf in valid.groupby("model"):
        for cwe_group, gdf in mdf.groupby("gt_cwe"):
            ps = (
                gdf.groupby("sample_id")
                .agg(gt_label=("gt_label", "first"), pred_label=("pred_label", _majority_label))
                .reset_index()
            )
            y_true, y_pred = ps["gt_label"].tolist(), ps["pred_label"].tolist()
            t2.add_row(
                model_name, cwe_group,
                f"{accuracy_score(y_true, y_pred):.3f}",
                f"{f1_score(y_true, y_pred, zero_division=0):.3f}",
                str(len(ps)),
            )
    _console.print(t2)

    # ── 3. CWE-ID prediction ──────────────────────────────────────────────────
    _console.rule("[bold]CWE-ID PREDICTION[/bold]  (vulnerable samples only)")
    t3 = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t3.add_column("Model")
    t3.add_column("CWE match rate", justify="right")
    t3.add_column("correct/total",  justify="right")

    vuln_valid = valid[(valid["gt_label"] == 1) & valid["pred_cwe"].notna()]
    for model_name, mdf in vuln_valid.groupby("model"):
        ps = (
            mdf.groupby("sample_id")
            .agg(
                gt_cwe   = ("gt_cwe",   "first"),
                pred_cwe = ("pred_cwe", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
            )
            .reset_index()
        )
        correct = (ps["gt_cwe"] == ps["pred_cwe"]).mean()
        t3.add_row(model_name, f"{correct:.1%}", f"{int(correct * len(ps))}/{len(ps)}")
    _console.print(t3)

    # ── 4. Self-consistency ───────────────────────────────────────────────────
    _console.rule("[bold]SELF-CONSISTENCY[/bold]")
    t4 = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t4.add_column("Model")
    t4.add_column("Consistent", justify="right")
    t4.add_column("Inconsistent", justify="right")

    for model_name, mdf in valid.groupby("model"):
        ps         = mdf.groupby("sample_id")["pred_label"].apply(list)
        consistent = ps.apply(lambda p: len(set(p)) == 1)
        n_incon    = (~consistent).sum()
        t4.add_row(
            model_name,
            f"{consistent.mean():.1%}  ({consistent.sum()}/{len(consistent)})",
            str(n_incon),
        )
    _console.print(t4)

    # ── 5. Error rate ─────────────────────────────────────────────────────────
    _console.rule("[bold]PARSE / API ERROR RATE[/bold]")
    t5 = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    t5.add_column("Model")
    t5.add_column("Error rate", justify="right")
    t5.add_column("Fallback used", justify="right")

    for model_name, mdf in df.groupby("model"):
        t5.add_row(
            model_name,
            f"{mdf['pred_label'].isna().mean():.1%}",
            f"{(mdf['used_provider'] != mdf['model']).sum()}x",
        )
    _console.print(t5)
    _console.print()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
FEAST pipeline CLI
==================
Download, synthesize, and materialize vulnerability datasets.

Stage 0 operation:
  download        -- fetch all datasets to data/raw/

Stage 2 operations:
  processed/      -- per-dataset CWE-filtered parquets
  merged/         -- deduplicated, cross-source parquets

Stage 3 operation:
  materialized/   -- individual source files for static analysis tools

Usage:
  python main.py download                                     # Stage 0: fetch all datasets
  python main.py synthesize                                   # all langs, all sources, default CWE filter
  python main.py synthesize --lang python                     # Python only
  python main.py synthesize --lang python --sources "CVEfixes(Python),PyVul"
  python main.py synthesize --cwe-types leaf                  # leaf CWEs only
  python main.py synthesize --cwes CWE-79,CWE-89,CWE-22      # specific CWE whitelist
  python main.py synthesize --min-cwe-count 20                # drop sparse CWEs post-merge
  python main.py synthesize --branches real,synth             # exclude AI-generated
  python main.py materialize                                  # write source files from merged parquets
  python main.py materialize --lang python                    # Python only
  python main.py list                                         # show all sources
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
RAW_DIR    = ROOT / 'data' / 'raw'
PROC_DIR   = ROOT / 'data' / 'processed'
MERGED_DIR = ROOT / 'data' / 'merged'
MAT_DIR    = ROOT / 'data' / 'materialized'
CWE_XML    = ROOT / 'data' / 'cwec_latest.xml'

_LANG_EXT = {'C/C++': '.c', 'Java': '.java', 'Python': '.py'}

BRANCH_PRIORITY = {'real': 0, 'synth': 1, 'ai': 2}
_VALID_BRANCHES = frozenset(BRANCH_PRIORITY)
_VALID_CWE_TYPES = frozenset({'leaf', 'non-leaf', 'category', 'deprecated', 'unknown'})
_LANG_SLUG    = {'C/C++': 'c_cpp', 'Java': 'java', 'Python': 'python'}
_LANG_ALIASES = {
    'c': 'C/C++', 'cpp': 'C/C++', 'c/c++': 'C/C++',
    'java': 'Java',
    'python': 'Python', 'py': 'Python',
}
ALL_LANGS = list(_LANG_SLUG)

# ── rich ───────────────────────────────────────────────────────────────────────
from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich.tree    import Tree
from rich         import box

console = Console(highlight=False)

# ── CWE navigator (lazy singleton) ────────────────────────────────────────────
_nav        = None
_parent_ids: set[str] = set()


def _ensure_nav() -> None:
    global _nav, _parent_ids
    if _nav is not None:
        return
    if not CWE_XML.exists():
        _download_cwe_xml()
    from ingestion.cwe_navigator import CWENavigator
    _nav = CWENavigator(str(CWE_XML))
    _parent_ids = {p for parents in _nav.child_of.values() for p in parents}
    console.print(f'  CWE XML: {len(_nav.weaknesses):,} weaknesses loaded', style='dim')


def _download_cwe_xml() -> None:
    import io, urllib.request, zipfile
    console.print('Downloading CWE XML from MITRE...', style='dim')
    url = 'https://cwe.mitre.org/data/xml/cwec_latest.xml.zip'
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        name = next(n for n in zf.namelist() if n.endswith('.xml'))
        CWE_XML.parent.mkdir(parents=True, exist_ok=True)
        CWE_XML.write_bytes(zf.read(name))
    console.print(f'  Saved {CWE_XML.name} ({CWE_XML.stat().st_size / 1e6:.1f} MB)', style='green')


def _cwe_type(cwe: str) -> str:
    num = cwe.removeprefix('CWE-').strip()
    if num in _nav.categories or num in _nav.views:
        return 'category'
    if num not in _nav.weaknesses:
        return 'unknown'
    if _nav.get_element_name(num).startswith('DEPRECATED:'):
        return 'deprecated'
    return 'leaf' if num not in _parent_ids else 'non-leaf'


# ── helpers ────────────────────────────────────────────────────────────────────

def _norm(code: str) -> str:
    code = code.replace('\r\n', '\n').replace('\r', '\n')
    lines = [l.rstrip() for l in code.split('\n')]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)


def _hash(code: str) -> str:
    return hashlib.sha256(_norm(code).encode()).hexdigest()


def _slug(name: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _csv_items(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(',') if item.strip()]


def _parse_cwe_types(raw: str) -> set[str]:
    if raw.lower().strip() == 'all':
        return set(_VALID_CWE_TYPES)
    values = {item.lower() for item in _csv_items(raw)}
    unknown = values - _VALID_CWE_TYPES
    if unknown:
        raise ValueError(
            f'unknown CWE type(s): {", ".join(sorted(unknown))}; '
            f'choose: {", ".join(sorted(_VALID_CWE_TYPES))}, all'
        )
    if not values:
        raise ValueError('at least one CWE type is required')
    return values


def _parse_cwe_whitelist(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    cwes: set[str] = set()
    invalid: list[str] = []
    for item in _csv_items(raw):
        m = re.fullmatch(r'(?:CWE-?)?(\d+)', item, flags=re.IGNORECASE)
        if m:
            cwes.add(f'CWE-{int(m.group(1))}')
        else:
            invalid.append(item)
    if invalid:
        raise ValueError(f'invalid CWE ID(s): {", ".join(invalid)}')
    if not cwes:
        raise ValueError('at least one CWE ID is required')
    return cwes


def _parse_branches(raw: str | None) -> set[str] | None:
    if not raw or raw.lower().strip() == 'all':
        return None
    values = {item.lower() for item in _csv_items(raw)}
    unknown = values - _VALID_BRANCHES
    if unknown:
        raise ValueError(
            f'unknown branch(es): {", ".join(sorted(unknown))}; '
            f'choose: {", ".join(sorted(_VALID_BRANCHES))}, all'
        )
    if not values:
        raise ValueError('at least one branch is required')
    return values


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        try:
            target.relative_to(dest)
        except ValueError as exc:
            raise ValueError(f'unsafe ZIP member path: {member.filename}') from exc
    zf.extractall(dest)


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tf.getmembers():
        if member.issym() or member.islnk():
            raise ValueError(f'unsafe TAR link member: {member.name}')
        target = (dest / member.name).resolve()
        try:
            target.relative_to(dest)
        except ValueError as exc:
            raise ValueError(f'unsafe TAR member path: {member.name}') from exc
    tf.extractall(dest)


def _resolve_source(name: str, registry: dict) -> str | None:
    """Case-insensitive / slug-based lookup into a registry dict."""
    if name in registry:
        return name
    low = name.lower()
    for k in registry:
        if k.lower() == low:
            return k
    s = _slug(name)
    for k in registry:
        if _slug(k) == s:
            return k
    return None


# ── ID assignment ──────────────────────────────────────────────────────────────

def _with_ids(loader):
    """Wrap a loader so every FunctionSample gets a stable content-derived sample_id."""
    def _wrapped():
        samples = loader()
        for s in samples:
            if not s.sample_id:
                s.sample_id = _hash(s.code)[:16]
        return samples
    return _wrapped


# ── dataset registry ───────────────────────────────────────────────────────────

def _build_registry(raw: Path) -> dict[str, dict[str, object]]:
    """Return {language: {dataset_name: loader_fn}} for all sources."""
    from ingestion.primevul        import extract_primevul
    from ingestion.icvul           import extract_icvul
    from ingestion.cvefixes        import extract_cvefixes
    from ingestion.megavul         import extract_megavul
    from ingestion.secvuleval      import extract_secvuleval
    from ingestion.crossvul        import extract_crossvul
    from ingestion.sven            import extract_sven
    from ingestion.juliet          import extract_juliet
    from ingestion.castle          import extract_castle
    from ingestion.llmseceval      import extract_llmseceval
    from ingestion.owasp_benchmark import extract_owasp_benchmark
    from ingestion.capec_llm       import extract_capec_llm
    from ingestion.patcheval       import extract_patcheval
    from ingestion.pyvul           import extract_pyvul
    from ingestion.security_eval   import extract_security_eval

    def _icvul():
        csv = next((raw / 'icvul').rglob('function_info.csv'), None)
        if csv is None:
            raise FileNotFoundError(f'function_info.csv not found under {raw}/icvul')
        return extract_icvul(csv.parent)

    raw_registry = {
        'C/C++': {
            'PrimeVul':       lambda: (extract_primevul(raw / 'primevul_train.jsonl') +
                                       extract_primevul(raw / 'primevul_test.jsonl')),
            'ICVul':          _icvul,
            'CVEfixes(C)':    lambda: extract_cvefixes(raw / 'cvefixes',         language='C'),
            'MegaVul':        lambda: extract_megavul(raw / 'megavul'),
            'SecVulEval':     lambda: extract_secvuleval(raw / 'secvuleval.csv'),
            'CrossVul(C)':    lambda: extract_crossvul(raw / 'crossvul.zip',     language='C/C++'),
            'SVEN(C)':        lambda: extract_sven(raw / 'sven',                 language='C/C++'),
            'Juliet(C)':      lambda: extract_juliet(raw / 'juliet_c.zip',       language='C/C++'),
            'CASTLE':         lambda: extract_castle(raw / 'castle' / 'datasets'),
            'LLMSecEval(C)':  lambda: extract_llmseceval(raw / 'llmseceval',     language='C/C++'),
        },
        'Java': {
            'CVEfixes(Java)':   lambda: extract_cvefixes(raw / 'cvefixes',               language='Java'),
            'CrossVul(Java)':   lambda: extract_crossvul(raw / 'crossvul.zip',           language='Java'),
            'Juliet(Java)':     lambda: extract_juliet(raw / 'juliet_java.zip',           language='Java'),
            'OWASP(Java)':      lambda: extract_owasp_benchmark(raw / 'owasp_benchmark', language='Java'),
            'CAPEC_LLM(Java)':  lambda: extract_capec_llm(raw / 'capec_llm',             language='Java'),
        },
        'Python': {
            'CVEfixes(Python)':  lambda: extract_cvefixes(raw / 'cvefixes',                      language='Python'),
            'PatchEval':         lambda: extract_patcheval(raw / 'patcheval'),
            'CrossVul(Python)':  lambda: extract_crossvul(raw / 'crossvul.zip',                  language='Python'),
            'PyVul':             lambda: extract_pyvul(raw / 'pyvul'),
            'SVEN(Python)':      lambda: extract_sven(raw / 'sven',                              language='Python'),
            'OWASP(Python)':     lambda: extract_owasp_benchmark(raw / 'owasp_benchmark_python', language='Python'),
            'LLMSecEval':        lambda: extract_llmseceval(raw / 'llmseceval',                  language='Python'),
            'SecurityEval':      lambda: extract_security_eval(raw / 'security_eval'),
            'CAPEC_LLM(Python)': lambda: extract_capec_llm(raw / 'capec_llm',                   language='Python'),
        },
    }
    return {
        lang: {name: _with_ids(loader) for name, loader in sources.items()}
        for lang, sources in raw_registry.items()
    }


# ── loading ────────────────────────────────────────────────────────────────────

def _load_collections(
    registry: dict,
    sources: list[str] | None,
    branches: set[str] | None,
) -> dict[str, list]:
    names = sources or list(registry)
    out: dict[str, list] = {}
    for name in names:
        resolved = _resolve_source(name, registry)
        if resolved is None:
            console.print(f'  [red]✗[/red] {name}  — unknown (run `list` to see available sources)')
            continue
        try:
            samples = registry[resolved]()
            if branches:
                samples = [s for s in samples if s.branch in branches]
            if not samples:
                console.print(f'  [yellow]~[/yellow] {resolved:<26}  0 samples after branch filter')
            else:
                out[resolved] = samples
                console.print(f'  [green]✓[/green] {resolved:<26} {len(samples):>8,} samples')
        except FileNotFoundError as e:
            console.print(f'  [yellow]~[/yellow] {resolved:<26}  SKIPPED  [dim]{e}[/dim]')
    return out


# ── per-dataset processing ─────────────────────────────────────────────────────

def _to_df(
    name: str,
    samples: list,
    language: str,
    allowed_types: set[str],
    allowed_cwes: set[str] | None,
) -> pd.DataFrame:
    """CWE-filter samples and convert to DataFrame."""
    rows = []
    for s in samples:
        if s.label == 1:
            valid = [
                c for c in s.cwes
                if _cwe_type(c) in allowed_types
                and (allowed_cwes is None or c in allowed_cwes)
            ]
            if not valid:
                continue
        else:
            valid = []
        h = _hash(s.code)
        rows.append({
            'code': s.code, 'language': language, 'label': s.label,
            'cwes': valid,  'branch': s.branch,   'source': name,
            'code_hash': h,
            'sample_id': s.sample_id or h[:16],
        })
    if not rows:
        return pd.DataFrame(columns=['code', 'language', 'label', 'cwes', 'branch', 'source', 'code_hash', 'sample_id'])
    return pd.DataFrame(rows)


def _process_lang(
    collections: dict[str, list],
    language: str,
    allowed_types: set[str],
    allowed_cwes: set[str] | None,
) -> dict[str, pd.DataFrame]:
    """Save per-dataset processed parquets; return {name: DataFrame}."""
    lang_dir = PROC_DIR / _LANG_SLUG[language]
    lang_dir.mkdir(parents=True, exist_ok=True)

    t = Table(box=box.SIMPLE, show_header=True, header_style='bold dim', pad_edge=False)
    t.add_column('Source',    style='cyan', no_wrap=True, min_width=26)
    t.add_column('Total',     justify='right', min_width=8)
    t.add_column('Vuln',      justify='right', style='red',   min_width=7)
    t.add_column('Safe',      justify='right', style='green', min_width=7)
    t.add_column('File',      style='dim')

    dfs: dict[str, pd.DataFrame] = {}
    for name, samples in collections.items():
        df = _to_df(name, samples, language, allowed_types, allowed_cwes)
        out = lang_dir / f'{_slug(name)}.parquet'
        df.to_parquet(out, index=False)
        n_v = int((df['label'] == 1).sum())
        n_s = int((df['label'] == 0).sum())
        t.add_row(name, f'{len(df):,}', f'{n_v:,}', f'{n_s:,}',
                  f'processed/{_LANG_SLUG[language]}/{out.name}')
        dfs[name] = df

    console.print(t)
    return dfs


# ── merging ────────────────────────────────────────────────────────────────────

def _merge_lang(
    dfs: dict[str, pd.DataFrame],
    language: str,
    min_cwe_count: int,
) -> pd.DataFrame:
    """Dedup by code hash, apply optional CWE count threshold, save merged parquet."""
    non_empty = {k: v for k, v in dfs.items() if len(v) > 0}
    if not non_empty:
        console.print(f'  [yellow]Nothing to merge for {language}[/yellow]')
        return pd.DataFrame()

    combined = pd.concat(list(non_empty.values()), ignore_index=True)
    combined['_prio'] = combined['branch'].map(lambda b: BRANCH_PRIORITY.get(b, 99))
    combined = combined.sort_values('_prio').reset_index(drop=True)

    label_counts = combined.groupby('code_hash', sort=False)['label'].nunique()
    n_conflicts = int((label_counts > 1).sum())
    if n_conflicts:
        console.print(f'  [yellow]![/yellow]  Label conflicts: {n_conflicts} hashes -- first (higher priority) kept')

    df = (combined.drop_duplicates(subset='code_hash', keep='first')
                  .drop(columns=['_prio'])
                  .reset_index(drop=True))
    n_dup = len(combined) - len(df)

    # optional: drop CWEs below sample-count threshold
    n_cwes_dropped = 0
    n_rows_dropped = 0
    if min_cwe_count > 1:
        cwe_counts: Counter = Counter(
            c for cwes in df.loc[df['label'] == 1, 'cwes'] for c in cwes
        )
        allowed_by_count = {c for c, n in cwe_counts.items() if n >= min_cwe_count}
        n_cwes_dropped = len(cwe_counts) - len(allowed_by_count)

        df = df.copy()
        vuln_mask = df['label'] == 1
        df.loc[vuln_mask, 'cwes'] = df.loc[vuln_mask, 'cwes'].map(
            lambda cwes: [c for c in cwes if c in allowed_by_count]
        )
        before = len(df)
        df = df[~((df['label'] == 1) & (df['cwes'].apply(len) == 0))].reset_index(drop=True)
        n_rows_dropped = before - len(df)

    out = MERGED_DIR / f'{_LANG_SLUG[language]}_merged.parquet'
    df.to_parquet(out, index=False)

    n_vuln    = int((df['label'] == 1).sum())
    n_safe    = int((df['label'] == 0).sum())
    all_cwes  = {c for cwes in df['cwes'] for c in cwes}

    t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    t.add_column(style='dim', no_wrap=True, min_width=30)
    t.add_column(justify='right')
    t.add_row('Input (after CWE filter)',     f'{len(combined):,}')
    t.add_row('Duplicates removed',           f'{n_dup:,}')
    if n_cwes_dropped:
        t.add_row(f'CWEs below min-count ({min_cwe_count})', f'{n_cwes_dropped} CWEs / {n_rows_dropped} samples dropped')
    t.add_row('[bold]Merged total[/bold]',
              f'[bold]{len(df):,}[/bold]  ({n_vuln:,} vuln + {n_safe:,} safe)')
    t.add_row('Unique CWEs',                  str(len(all_cwes)))
    t.add_row('[green]Saved[/green]',         _display_path(out))
    console.print(t)
    return df


# ── synthesize command ─────────────────────────────────────────────────────────

def cmd_synthesize(args) -> None:
    # resolve language(s)
    if args.lang.lower() == 'all':
        langs = ALL_LANGS
    else:
        key = args.lang.lower()
        if key not in _LANG_ALIASES:
            console.print(f'[red]Unknown language:[/red] {args.lang!r}  — choose: c, java, python, all')
            sys.exit(1)
        langs = [_LANG_ALIASES[key]]

    # CWE type filter
    try:
        allowed_types = _parse_cwe_types(args.cwe_types)
    except ValueError as e:
        console.print(f'[red]Invalid --cwe-types:[/red] {e}')
        sys.exit(1)

    # specific CWE whitelist
    try:
        allowed_cwes = _parse_cwe_whitelist(args.cwes)
    except ValueError as e:
        console.print(f'[red]Invalid --cwes:[/red] {e}')
        sys.exit(1)

    # branch filter
    try:
        branches = _parse_branches(args.branches)
    except ValueError as e:
        console.print(f'[red]Invalid --branches:[/red] {e}')
        sys.exit(1)

    if args.min_cwe_count < 1:
        console.print('[red]Invalid --min-cwe-count:[/red] value must be >= 1')
        sys.exit(1)

    # source filter
    sources_filter: list[str] | None = None
    if args.sources:
        sources_filter = [s.strip() for s in args.sources.split(',')]

    _ensure_nav()

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    MERGED_DIR.mkdir(parents=True, exist_ok=True)

    registry = _build_registry(RAW_DIR)
    summary: list[tuple[str, int, int, int]] = []

    for language in langs:
        subtitle_parts = [f'cwe-types=[cyan]{args.cwe_types}[/cyan]']
        if args.cwes:
            subtitle_parts.append(f'cwes=[cyan]{args.cwes}[/cyan]')
        if args.min_cwe_count > 1:
            subtitle_parts.append(f'min-cwe-count=[cyan]{args.min_cwe_count}[/cyan]')
        subtitle_parts.append(f'branches=[cyan]{args.branches}[/cyan]')

        console.print()
        console.print(Panel(
            f'[bold]{language}[/bold]',
            subtitle='  '.join(subtitle_parts),
            expand=False,
        ))

        console.print('[bold]Loading[/bold]')
        collections = _load_collections(registry[language], sources_filter, branches)
        if not collections:
            console.print(f'  [red]No data for {language} — check data/raw/[/red]')
            continue

        console.print('\n[bold]Processing  →  processed/[/bold]')
        dfs = _process_lang(collections, language, allowed_types, allowed_cwes)

        console.print('[bold]Merging  →  merged/[/bold]')
        df = _merge_lang(dfs, language, args.min_cwe_count)
        if len(df):
            summary.append((
                language, len(df),
                int((df['label'] == 1).sum()),
                int((df['label'] == 0).sum()),
            ))

    if len(summary) > 1:
        console.print()
        t = Table(title='[bold]Summary[/bold]', box=box.ROUNDED, show_lines=True)
        t.add_column('Language', style='bold',  no_wrap=True)
        t.add_column('Total',    justify='right')
        t.add_column('Vuln',     justify='right', style='red')
        t.add_column('Safe',     justify='right', style='green')
        for lang, total, vuln, safe in summary:
            t.add_row(lang, f'{total:,}', f'{vuln:,}', f'{safe:,}')
        console.print(t)


# ── list command ───────────────────────────────────────────────────────────────

def cmd_list(_args) -> None:
    registry = _build_registry(RAW_DIR)

    tree = Tree('[bold]FEAST source datasets[/bold]')
    for lang, sources in registry.items():
        branch = tree.add(f'[bold cyan]{lang}[/bold cyan]  [dim]({len(sources)} sources)[/dim]')
        for name in sources:
            branch.add(name)
    console.print(tree)

    console.print()
    console.print('[dim]Tip: pass --sources to select a subset, e.g.[/dim]')
    console.print('[dim]  --sources "CVEfixes(Python),PyVul,LLMSecEval"[/dim]')
    console.print('[dim]  --sources primevul,icvul          (slug-style also accepted)[/dim]')


# ── materialize command ────────────────────────────────────────────────────────

def cmd_materialize(args) -> None:
    """Stage 3: write merged samples as individual source files."""
    if args.lang.lower() == 'all':
        langs = ALL_LANGS
    else:
        key = args.lang.lower()
        if key not in _LANG_ALIASES:
            console.print(f'[red]Unknown language:[/red] {args.lang!r}  -- choose: c, java, python, all')
            sys.exit(1)
        langs = [_LANG_ALIASES[key]]

    total_written = 0

    for language in langs:
        ext    = _LANG_EXT[language]
        slug   = _LANG_SLUG[language]
        parq   = MERGED_DIR / f'{slug}_merged.parquet'

        if not parq.exists():
            console.print(f'[yellow]~[/yellow]  {language}: {parq} not found — run synthesize first')
            continue

        df = pd.read_parquet(parq)
        if 'sample_id' not in df.columns:
            df['sample_id'] = df['code_hash'].str[:16]
        if 'source' not in df.columns:
            console.print(f'[red]✗[/red]  {language}: merged parquet missing "source" column')
            continue

        lang_dir = MAT_DIR / slug
        n_written = 0
        n_skip    = 0

        console.print()
        console.print(Panel(f'[bold]{language}[/bold]  [dim]({len(df):,} samples)[/dim]', expand=False))

        for row in df[['source', 'sample_id', 'code']].itertuples(index=False):
            dataset_dir = lang_dir / _slug(row.source)
            dataset_dir.mkdir(parents=True, exist_ok=True)
            out = dataset_dir / f'{row.sample_id}{ext}'
            if not args.overwrite and out.exists():
                n_skip += 1
                continue
            out.write_text(row.code, encoding='utf-8')
            n_written += 1

        # write per-language index parquet
        index_path = lang_dir / 'index.parquet'
        df[['sample_id', 'source', 'label', 'cwes', 'branch', 'code_hash']].to_parquet(index_path, index=False)

        t = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        t.add_column(style='dim', no_wrap=True, min_width=24)
        t.add_column(justify='right')
        t.add_row('Files written',   f'{n_written:,}')
        if n_skip:
            t.add_row('Skipped (exist)', f'{n_skip:,}')
        t.add_row('[green]Index saved[/green]', _display_path(index_path))
        console.print(t)
        total_written += n_written

    console.print()
    console.print(f'[bold green]Done.[/bold green]  {total_written:,} file(s) written  ->  data/materialized/')


# ── download command ───────────────────────────────────────────────────────────

def cmd_download(_args) -> None:
    """Stage 0: download all 16 datasets to data/raw/."""
    import re as _re
    import shutil
    import subprocess
    import tarfile
    import urllib.request
    import zipfile as _zipfile

    raw = RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)

    def _present(path: Path) -> bool:
        return path.exists() and (path.is_file() or any(path.iterdir()))

    def _ok(label: str, detail: str = '') -> None:
        console.print(f'  [green]✓[/green]  {label:<28}{detail}')

    def _skip(label: str) -> None:
        console.print(f'  [dim]~[/dim]  {label:<28}already present -- skipped')

    def _err(label: str, msg: str) -> None:
        console.print(f'  [red]✗[/red]  {label:<28}{msg}')

    def _manual(label: str, dest: str, note: str) -> None:
        console.print(f'  [yellow]![/yellow]  {label:<28}[bold]manual download required[/bold]')
        console.print(f'       Download from: {note}')
        console.print(f'       Place at:      data/raw/{dest}')

    def _git_clone(url: str, dest: Path, label: str) -> None:
        if _present(dest):
            _skip(label)
            return
        dest.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(['git', 'clone', '--depth=1', url, str(dest)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            _err(label, r.stderr.strip())
        else:
            _ok(label, f'{sum(1 for _ in dest.rglob("*") if _.is_file())} files')

    def _hf_files(repo_id: str, files: list[str], dest: Path, label: str) -> None:
        if _present(dest):
            _skip(label)
            return
        from huggingface_hub import hf_hub_download
        dest.mkdir(parents=True, exist_ok=True)
        for hf_path in files:
            local = hf_hub_download(repo_id=repo_id, filename=hf_path, repo_type='dataset')
            shutil.copy(local, dest / Path(hf_path).name)
        _ok(label, f'{len(files)} parquet(s)')

    def _hf_snapshot(repo_id: str, dest: Path, label: str) -> None:
        if _present(dest):
            _skip(label)
            return
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=repo_id, repo_type='dataset')
        shutil.copytree(local, dest, dirs_exist_ok=True)
        _ok(label, f'{sum(1 for _ in dest.rglob("*") if _.is_file())} files')

    def _urllib_get(url: str, dest: Path, label: str) -> None:
        if _present(dest):
            _skip(label)
            return
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, dest)
        _ok(label, f'{dest.stat().st_size / 1e6:.0f} MB')

    console.print(Panel('[bold]Stage 0 -- Download[/bold]  (data/raw/)', expand=False))

    # 1. PrimeVul  (HuggingFace -- two JSONL files)
    pv_train, pv_test = raw / 'primevul_train.jsonl', raw / 'primevul_test.jsonl'
    if pv_train.exists() and pv_test.exists():
        _skip('PrimeVul')
    else:
        from huggingface_hub import hf_hub_download
        for split in ['primevul_train.jsonl', 'primevul_test.jsonl']:
            local = hf_hub_download(repo_id='starsofchance/PrimeVul', filename=split, repo_type='dataset')
            shutil.copy(local, raw / split)
        _ok('PrimeVul', 'train + test')

    # 2. ICVul  (Google Drive -- requires gdown)
    icvul_dir = raw / 'icvul'
    if _present(icvul_dir):
        _skip('ICVul')
    else:
        import gdown
        icvul_dir.mkdir(parents=True, exist_ok=True)
        archive = raw / '_icvul_archive'
        downloaded = gdown.download(id='1Bnnb7kJa8GEfyESIAuGXj2z0g8FvXgRk', output=str(archive))
        if downloaded is None:
            _err('ICVul', 'gdown failed -- check file permissions')
        else:
            arc = next(raw.glob('_icvul_archive*'))
            if _zipfile.is_zipfile(arc):
                with _zipfile.ZipFile(arc) as zf:
                    _safe_extract_zip(zf, icvul_dir)
            elif tarfile.is_tarfile(arc):
                with tarfile.open(arc) as tf:
                    _safe_extract_tar(tf, icvul_dir)
            arc.unlink(missing_ok=True)
            _ok('ICVul', 'extracted')

    # 3. CVEfixes  (HuggingFace -- 3 parquet shards)
    _hf_files('hitoshura25/cvefixes',
              ['data/train-00000-of-00003.parquet',
               'data/train-00001-of-00003.parquet',
               'data/train-00002-of-00003.parquet'],
              raw / 'cvefixes', 'CVEfixes')

    # 4. MegaVul  (HuggingFace -- 2 parquet shards)
    _hf_files('hitoshura25/megavul',
              ['data/train-00000-of-00002.parquet',
               'data/train-00001-of-00002.parquet'],
              raw / 'megavul', 'MegaVul')

    # 5. SecVulEval  (HuggingFace datasets library)
    secvuleval = raw / 'secvuleval.csv'
    if secvuleval.exists():
        _skip('SecVulEval')
    else:
        from datasets import load_dataset
        ds = load_dataset('arag0rn/SecVulEval', trust_remote_code=True)
        df = pd.concat([ds[split].to_pandas() for split in ds], ignore_index=True)
        df.to_csv(secvuleval, index=False)
        _ok('SecVulEval', f'{len(df):,} rows')

    # 6. CrossVul  (Zenodo -- MANUAL)
    crossvul = raw / 'crossvul.zip'
    if crossvul.exists():
        _skip('CrossVul')
    else:
        _manual('CrossVul', 'crossvul.zip', 'Zenodo (search "CrossVul" dataset)')

    # 7. SVEN  (HuggingFace)
    _hf_snapshot('bstee615/sven', raw / 'sven', 'SVEN')

    # 8-9. Juliet C/C++ and Java  (NIST SARD)
    _urllib_get(
        'https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip',
        raw / 'juliet_c.zip', 'Juliet C/C++',
    )
    _urllib_get(
        'https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-java-v1-3.zip',
        raw / 'juliet_java.zip', 'Juliet Java',
    )

    # 10. CASTLE  (GitHub)
    _git_clone('https://github.com/CASTLE-Benchmark/CASTLE-Benchmark.git', raw / 'castle', 'CASTLE')

    # 11. LLMSecEval  (Zenodo ZIP -- MANUAL; safe samples from GitHub -- automatic)
    llmsec_dir  = raw / 'llmseceval'
    zenodo_dir  = llmsec_dir / 'zenodo'
    copilot_zip = raw / 'copilot-cwe-scenarios-dataset.zip'
    zenodo_ok   = zenodo_dir.exists() and any(zenodo_dir.rglob('gen_scenario/*.py'))
    secure_ok   = llmsec_dir.exists() and any(llmsec_dir.glob('CWE-*/Secure/*.py'))

    if zenodo_ok:
        _skip('LLMSecEval (vuln)')
    elif copilot_zip.exists():
        zenodo_dir.mkdir(parents=True, exist_ok=True)
        with _zipfile.ZipFile(copilot_zip) as zf:
            _safe_extract_zip(zf, zenodo_dir)
        _ok('LLMSecEval (vuln)', 'extracted from ZIP')
    else:
        _manual('LLMSecEval (vuln)', 'copilot-cwe-scenarios-dataset.zip',
                'Zenodo record 5225651')

    if secure_ok:
        _skip('LLMSecEval (safe)')
    else:
        github_clone = raw / '_llmseceval_github'
        _git_clone('https://github.com/tuhh-softsec/LLMSecEval.git', github_clone, 'LLMSecEval (safe)')
        secure_src = github_clone / 'Dataset' / 'Secure Code Samples'
        if secure_src.exists():
            llmsec_dir.mkdir(parents=True, exist_ok=True)
            n = 0
            for cwe_sub in secure_src.iterdir():
                if not cwe_sub.is_dir():
                    continue
                m = _re.search(r'\d+', cwe_sub.name)
                if not m:
                    continue
                dest_sec = llmsec_dir / f'CWE-{int(m.group())}' / 'Secure'
                dest_sec.mkdir(parents=True, exist_ok=True)
                for f in cwe_sub.iterdir():
                    if f.is_file():
                        shutil.copy(f, dest_sec / f.name)
                        n += 1
            shutil.rmtree(github_clone, ignore_errors=True)
            _ok('LLMSecEval (safe)', f'{n} files')

    # 12. OWASP Benchmark  (GitHub x2)
    _git_clone('https://github.com/OWASP-Benchmark/BenchmarkJava.git',
               raw / 'owasp_benchmark', 'OWASP (Java)')
    _git_clone('https://github.com/OWASP-Benchmark/BenchmarkPython.git',
               raw / 'owasp_benchmark_python', 'OWASP (Python)')

    # 13-16. GitHub repos
    _git_clone('https://github.com/llmForCapec/CAPECDatasetsLLM.git', raw / 'capec_llm',     'CAPEC_LLM')
    _git_clone('https://github.com/bytedance/PatchEval.git',           raw / 'patcheval',     'PatchEval')
    _git_clone('https://github.com/billquan/PyVul.git',                raw / 'pyvul',         'PyVul')
    _git_clone('https://github.com/s2e-lab/SecurityEval.git',          raw / 'security_eval', 'SecurityEval')

    # ── manual-download reminder ───────────────────────────────────────────────
    pending = []
    if not crossvul.exists():
        pending.append(('CrossVul',         'crossvul.zip',                    'Zenodo (search "CrossVul" dataset)'))
    if not zenodo_ok and not copilot_zip.exists():
        pending.append(('LLMSecEval (vuln)', 'copilot-cwe-scenarios-dataset.zip', 'Zenodo record 5225651'))

    console.print()
    if pending:
        lines = '\n\n'.join(
            f'[bold]{name}[/bold]\n  Download from: {note}\n  Save to:       data/raw/{dest}'
            for name, dest, note in pending
        )
        console.print(Panel(
            '[bold yellow]Manual downloads still required[/bold yellow]\n\n' + lines,
            expand=False,
        ))
    else:
        console.print('[bold green]All datasets present.[/bold green]  Run [cyan]synthesize[/cyan] next.')


# ── argument parser ────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python main.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='FEAST - vulnerability dataset pipeline CLI',
        epilog="""\
examples:
  python main.py download
  python main.py synthesize
  python main.py synthesize --lang python
  python main.py synthesize --lang python --sources "CVEfixes(Python),PyVul"
  python main.py synthesize --cwe-types leaf
  python main.py synthesize --cwes CWE-79,CWE-89,CWE-22
  python main.py synthesize --min-cwe-count 20
  python main.py synthesize --branches real,synth
  python main.py materialize
  python main.py materialize --lang python
  python main.py list
""",
    )
    sub = p.add_subparsers(dest='command', required=True)

    # ── download ───────────────────────────────────────────────────────────────
    sub.add_parser(
        'download', aliases=['dl', 'd'],
        help='Download all datasets  ->  data/raw/',
    )

    # ── synthesize ─────────────────────────────────────────────────────────────
    syn = sub.add_parser(
        'synthesize', aliases=['synth', 's'],
        help='CWE-filter sources + deduplicate  ->  processed/ and merged/',
    )
    syn.add_argument(
        '--lang', default='all', metavar='LANG',
        help='Language to process: c, java, python, all  [default: all]',
    )
    syn.add_argument(
        '--sources', default=None, metavar='SRC1,SRC2,...',
        help='Comma-separated source dataset names  [default: all available]',
    )
    syn.add_argument(
        '--cwe-types', default='leaf,non-leaf', metavar='TYPES', dest='cwe_types',
        help=(
            'CWE node types to keep for vulnerable samples:\n'
            '  leaf, non-leaf, category, deprecated, unknown, all\n'
            '  [default: leaf,non-leaf]'
        ),
    )
    syn.add_argument(
        '--cwes', default=None, metavar='CWE-79,CWE-89,...',
        help='Additional whitelist of specific CWE IDs  [default: all]',
    )
    syn.add_argument(
        '--min-cwe-count', type=int, default=1, metavar='N', dest='min_cwe_count',
        help='After merge, drop CWEs with fewer than N vulnerable samples  [default: 1 = off]',
    )
    syn.add_argument(
        '--branches', default='all', metavar='BRANCHES',
        help='Source branches to include: real, synth, ai, or comma-separated  [default: all]',
    )
    syn.add_argument(
        '--data-dir', type=Path, default=None, metavar='DIR', dest='data_dir',
        help='Override raw data directory  [default: data/raw/]',
    )

    # ── materialize ────────────────────────────────────────────────────────────
    mat = sub.add_parser(
        'materialize', aliases=['mat', 'm'],
        help='Write merged samples as source files  ->  materialized/',
    )
    mat.add_argument(
        '--lang', default='all', metavar='LANG',
        help='Language to materialize: c, java, python, all  [default: all]',
    )
    mat.add_argument(
        '--overwrite', action='store_true',
        help='Re-write files that already exist  [default: skip existing]',
    )

    # ── list ───────────────────────────────────────────────────────────────────
    sub.add_parser('list', aliases=['ls'], help='Show all available source datasets')

    return p


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    global RAW_DIR
    parser = _build_parser()
    args   = parser.parse_args()

    if hasattr(args, 'data_dir') and args.data_dir is not None:
        RAW_DIR = args.data_dir

    match args.command:
        case 'download' | 'dl' | 'd':
            cmd_download(args)
        case 'synthesize' | 'synth' | 's':
            cmd_synthesize(args)
        case 'materialize' | 'mat' | 'm':
            cmd_materialize(args)
        case 'list' | 'ls':
            cmd_list(args)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
FEAST pipeline CLI
==================
Synthesize and materialize vulnerability datasets with fine-grained filtering.

Stage 2 operations:
  processed/      -- per-dataset CWE-filtered parquets
  merged/         -- deduplicated, cross-source parquets

Stage 3 operation:
  materialized/   -- individual source files for static analysis tools

Usage:
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

    conflicts = combined.groupby('code_hash').filter(lambda g: g['label'].nunique() > 1)
    if len(conflicts):
        console.print(f'  [yellow]⚠[/yellow]  Label conflicts: {conflicts["code_hash"].nunique()} hashes — first (higher priority) kept')

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
        df['cwes'] = df.apply(
            lambda r: [c for c in r['cwes'] if c in allowed_by_count] if r['label'] == 1 else r['cwes'],
            axis=1,
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
    t.add_row('[green]Saved[/green]',         str(out.relative_to(ROOT)))
    console.print(t)
    return df


# ── synthesize command ─────────────────────────────────────────────────────────

def cmd_synthesize(args) -> None:
    _ensure_nav()

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
    if args.cwe_types.lower() == 'all':
        allowed_types = {'leaf', 'non-leaf', 'category', 'deprecated', 'unknown'}
    else:
        allowed_types = {t.strip() for t in args.cwe_types.split(',')}

    # specific CWE whitelist
    allowed_cwes: set[str] | None = None
    if args.cwes:
        raw_ids = [c.strip() for c in args.cwes.split(',')]
        allowed_cwes = {c if c.upper().startswith('CWE-') else f'CWE-{c}' for c in raw_ids}

    # branch filter
    branches: set[str] | None = None
    if args.branches and args.branches.lower() != 'all':
        branches = {b.strip() for b in args.branches.split(',')}

    # source filter
    sources_filter: list[str] | None = None
    if args.sources:
        sources_filter = [s.strip() for s in args.sources.split(',')]

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

        for _, row in df.iterrows():
            dataset_dir = lang_dir / _slug(row['source'])
            dataset_dir.mkdir(parents=True, exist_ok=True)
            out = dataset_dir / f'{row["sample_id"]}{ext}'
            if not args.overwrite and out.exists():
                n_skip += 1
                continue
            out.write_text(row['code'], encoding='utf-8')
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
        t.add_row('[green]Index saved[/green]', str(index_path.relative_to(ROOT)))
        console.print(t)
        total_written += n_written

    console.print()
    console.print(f'[bold green]Done.[/bold green]  {total_written:,} file(s) written  ->  data/materialized/')


# ── argument parser ────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python main.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='FEAST - vulnerability dataset pipeline CLI',
        epilog="""\
examples:
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
        case 'synthesize' | 'synth' | 's':
            cmd_synthesize(args)
        case 'materialize' | 'mat' | 'm':
            cmd_materialize(args)
        case 'list' | 'ls':
            cmd_list(args)


if __name__ == '__main__':
    main()

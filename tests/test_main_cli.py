import io
import tarfile
import zipfile
from types import SimpleNamespace

import pandas as pd
import pytest

import main as feast_cli


def test_parse_cwe_whitelist_normalises_ids():
    assert feast_cli._parse_cwe_whitelist("cwe-079,89,CWE89") == {"CWE-79", "CWE-89"}


def test_parse_cwe_whitelist_rejects_invalid_ids():
    with pytest.raises(ValueError):
        feast_cli._parse_cwe_whitelist("CWE-79,foo")


def test_parse_cwe_types_rejects_unknown_values():
    with pytest.raises(ValueError):
        feast_cli._parse_cwe_types("leaf,leef")


def test_parse_branches_normalises_and_rejects_unknown_values():
    assert feast_cli._parse_branches("Real,AI") == {"real", "ai"}
    with pytest.raises(ValueError):
        feast_cli._parse_branches("real,manual")


def test_path_component_handles_reserved_weird_and_long_names():
    reserved = feast_cli._path_component("CON", fallback="dataset")
    assert reserved != "CON"
    assert len(reserved) <= feast_cli._MAX_PATH_COMPONENT

    weird = feast_cli._path_component('..//bad:name*with?chars <>|"', fallback="dataset")
    assert weird
    assert ".." not in weird
    assert not any(ch in weird for ch in '<>:"/\\|?*')

    long_name = feast_cli._path_component("x" * 300, fallback="dataset", max_len=64)
    assert len(long_name) <= 64
    assert long_name.endswith(feast_cli._path_hash("x" * 300))


def test_safe_extract_zip_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../outside.txt", "nope")

    with zipfile.ZipFile(archive) as zf, pytest.raises(ValueError):
        feast_cli._safe_extract_zip(zf, tmp_path / "out")


def test_safe_extract_zip_rejects_windows_traversal_and_drive_paths(tmp_path):
    archive = tmp_path / "bad_windows.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(r"..\outside.txt", "nope")
        zf.writestr(r"C:\temp\outside.txt", "nope")

    with zipfile.ZipFile(archive) as zf, pytest.raises(ValueError):
        feast_cli._safe_extract_zip(zf, tmp_path / "out")


def test_safe_extract_zip_normalises_backslash_members(tmp_path):
    archive = tmp_path / "ok_windows.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr(r"nested\file.txt", "ok")

    out = tmp_path / "out"
    with zipfile.ZipFile(archive) as zf:
        feast_cli._safe_extract_zip(zf, out)

    assert (out / "nested" / "file.txt").read_text() == "ok"


def test_safe_extract_tar_rejects_path_traversal(tmp_path):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as tf:
        payload = b"nope"
        info = tarfile.TarInfo("../outside.txt")
        info.size = len(payload)
        tf.addfile(info, io.BytesIO(payload))

    data.seek(0)
    with tarfile.open(fileobj=data, mode="r") as tf, pytest.raises(ValueError):
        feast_cli._safe_extract_tar(tf, tmp_path / "out")


def test_merge_lang_keeps_priority_and_filters_sparse_cwes(tmp_path, monkeypatch):
    monkeypatch.setattr(feast_cli, "MERGED_DIR", tmp_path)

    vuln_real = pd.DataFrame([
        {
            "code": "def vuln_one(): pass",
            "language": "Python",
            "label": 1,
            "cwes": ["CWE-79"],
            "branch": "real",
            "source": "RealSet",
            "code_hash": "same",
            "sample_id": "same",
        },
        {
            "code": "def vuln_two(): pass",
            "language": "Python",
            "label": 1,
            "cwes": ["CWE-79"],
            "branch": "real",
            "source": "RealSet",
            "code_hash": "keep",
            "sample_id": "keep",
        },
        {
            "code": "def sparse(): pass",
            "language": "Python",
            "label": 1,
            "cwes": ["CWE-999"],
            "branch": "real",
            "source": "RealSet",
            "code_hash": "drop",
            "sample_id": "drop",
        },
    ])
    safe_ai_conflict = pd.DataFrame([
        {
            "code": "def vuln_one(): pass",
            "language": "Python",
            "label": 0,
            "cwes": [],
            "branch": "ai",
            "source": "AISet",
            "code_hash": "same",
            "sample_id": "same",
        }
    ])

    merged = feast_cli._merge_lang(
        {"RealSet": vuln_real, "AISet": safe_ai_conflict},
        "Python",
        min_cwe_count=2,
    )

    assert set(merged["code_hash"]) == {"same", "keep"}
    assert merged.loc[merged["code_hash"] == "same", "label"].item() == 1
    assert (tmp_path / "python_merged.parquet").exists()


def test_materialize_sanitises_weird_and_long_paths(tmp_path, monkeypatch):
    merged_dir = tmp_path / "merged"
    materialized_dir = tmp_path / "materialized"
    merged_dir.mkdir()
    monkeypatch.setattr(feast_cli, "MERGED_DIR", merged_dir)
    monkeypatch.setattr(feast_cli, "MAT_DIR", materialized_dir)

    long_sample_id = "CON" + ("x" * 300)
    weird_source = '..//CON:very weird dataset name with spaces and <>:"|?*' + ("z" * 200)
    code_hash = "a" * 64
    pd.DataFrame([
        {
            "code": "def safe():\n    return 1\n",
            "language": "Python",
            "label": 0,
            "cwes": [],
            "branch": "real",
            "source": weird_source,
            "code_hash": code_hash,
            "sample_id": long_sample_id,
        }
    ]).to_parquet(merged_dir / "python_merged.parquet", index=False)

    feast_cli.cmd_materialize(SimpleNamespace(lang="python", overwrite=True))

    index = pd.read_parquet(materialized_dir / "python" / "index.parquet")
    rel_path = index["materialized_path"].item()
    assert ".." not in rel_path.split("/")
    assert not any(ch in rel_path for ch in '<>:"\\|?*')
    assert all(len(part) <= feast_cli._MAX_PATH_COMPONENT + 3 for part in rel_path.split("/"))
    assert (materialized_dir / "python" / rel_path).exists()

import io
import tarfile
import zipfile

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


def test_safe_extract_zip_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../outside.txt", "nope")

    with zipfile.ZipFile(archive) as zf, pytest.raises(ValueError):
        feast_cli._safe_extract_zip(zf, tmp_path / "out")


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

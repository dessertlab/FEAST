import pandas as pd
import pytest

from analysis.canonical import CweCanonicalizer
from ingestion.cwe_navigator import CWENavigator

XML = "data/cwec_latest.xml"


@pytest.fixture(scope="module")
def navigator():
    return CWENavigator(XML)


def test_primary_parent_follows_view_1000_primary_edge(navigator):
    # CWE-120 has parents in views 700/1003/1340 too; only the 1000 primary (787) must win.
    assert navigator.primary_parent("120") == "787"
    assert navigator.primary_parent("CWE-798") == "1391"
    assert navigator.primary_parent("73") == "642"


def test_primary_path_reaches_pillar(navigator):
    assert navigator.primary_path("120") == ("120", "787", "119", "118", "664")
    assert navigator.abstraction("664") == "Pillar"


@pytest.mark.parametrize("level,expected", [
    ("pillar", {"79": "CWE-707", "89": "CWE-707", "120": "CWE-664", "125": "CWE-664"}),
    ("subcategory", {"79": "CWE-74", "89": "CWE-74", "120": "CWE-118", "125": "CWE-118"}),
    ("class", {"79": "CWE-74", "89": "CWE-943", "120": "CWE-119", "125": "CWE-119"}),
])
def test_family_mapping_per_level(navigator, level, expected):
    canon = CweCanonicalizer(navigator, level=level)
    for cwe, family in expected.items():
        assert canon.family(cwe) == family


def test_class_level_keeps_injection_distinct_but_merges_buffer(navigator):
    canon = CweCanonicalizer(navigator, level="class")
    # injection types stay separate, buffer over-read/copy merge
    assert canon.family("79") != canon.family("89")           # XSS vs SQLi
    assert canon.family("120") == canon.family("125") == "CWE-119"


def test_class_level_falls_back_to_pillar_child_when_no_class_exists(navigator):
    canon = CweCanonicalizer(navigator, level="class")
    # CWE-1024 is a Base weakness whose primary path is 1024 -> 697, with no Class node.
    assert navigator.abstraction("1024") == "Base"
    assert navigator.primary_path("1024") == ("1024", "697")
    assert canon.family("1024") == "CWE-1024"


def test_unmapped_tokens_return_none(navigator):
    canon = CweCanonicalizer(navigator, level="class")
    assert canon.family("CWE-0") is None
    assert canon.family("not-a-cwe") is None


def test_canonicalize_frame_rewrites_gt_and_tools(navigator):
    canon = CweCanonicalizer(navigator, level="class")
    df = pd.DataFrame([
        {"cwes": ["CWE-89"], "toolA": ["CWE-79"], "toolB": []},
        {"cwes": ["CWE-120", "CWE-125"], "toolA": ["CWE-787"], "toolB": ["CWE-0"]},
    ])
    out = canon.canonicalize_frame(df, ["toolA", "toolB"])
    assert out.at[0, "cwes"] == ["CWE-943"]
    assert out.at[0, "toolA"] == ["CWE-74"]
    assert out.at[1, "cwes"] == ["CWE-119"]   # 120 and 125 collapse to one family
    assert out.at[1, "toolA"] == ["CWE-119"]
    assert out.at[1, "toolB"] == []           # CWE-0 dropped as unmapped

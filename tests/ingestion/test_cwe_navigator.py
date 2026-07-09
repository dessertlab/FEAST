import xml.etree.ElementTree as ET

import pytest

from ingestion.cwe_navigator import CWENavigator


@pytest.fixture
def cwe_xml(tmp_path):
    path = tmp_path / "cwe.xml"
    path.write_text(
        """<?xml version="1.0"?>
<Weakness_Catalog xmlns="http://cwe.mitre.org/cwe-7">
  <Weaknesses>
    <Weakness ID="20" Name="Input Validation"/>
    <Weakness ID="77" Name="Command Injection">
      <Related_Weaknesses>
        <Related_Weakness Nature="PeerOf" CWE_ID="20"/>
      </Related_Weaknesses>
    </Weakness>
    <Weakness ID="79" Name="XSS">
      <Related_Weaknesses>
        <Related_Weakness Nature="ChildOf" CWE_ID="20"/>
      </Related_Weaknesses>
    </Weakness>
    <Weakness ID="89" Name="SQL Injection">
      <Related_Weaknesses>
        <Related_Weakness Nature="ChildOf" CWE_ID="79"/>
      </Related_Weaknesses>
    </Weakness>
    <Weakness ID="999" Name="Orphan"/>
  </Weaknesses>
  <Categories>
    <Category ID="1000" Name="Root Category">
      <Relationships>
        <Has_Member CWE_ID="20"/>
        <Has_Member/>
        <Has_Member CWE_ID="2000"/>
      </Relationships>
    </Category>
    <Category ID="2000" Name="Nested Category">
      <Relationships>
        <Has_Member CWE_ID="79"/>
      </Relationships>
    </Category>
    <Category ID="3000" Name="Detached Category">
      <Relationships>
        <Has_Member CWE_ID="999"/>
      </Relationships>
    </Category>
    <Category ID="4000" Name="Child Category">
      <Relationships>
        <Has_Member Nature="ChildOf" CWE_ID="1000"/>
      </Relationships>
    </Category>
  </Categories>
  <Views>
    <View ID="VIEW" Name="Research View">
      <Members>
        <Has_Member CWE_ID="1000"/>
        <Has_Member/>
      </Members>
    </View>
    <View ID="EMPTY" Name="Empty View">
      <Members/>
    </View>
  </Views>
</Weakness_Catalog>
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def navigator(cwe_xml):
    return CWENavigator(str(cwe_xml))


def test_navigator_parses_entities_and_child_relations(navigator):
    assert set(navigator.weaknesses) == {"20", "77", "79", "89", "999"}
    assert set(navigator.categories) == {"1000", "2000", "3000", "4000"}
    assert set(navigator.views) == {"VIEW", "EMPTY"}
    assert navigator.child_of["79"] == ["20"]
    assert navigator.child_of["4000"] == ["1000"]
    assert "77" not in navigator.child_of


def test_view_hierarchy_recurses_through_categories(navigator):
    assert navigator.view_structure["VIEW"]["VIEW"] == ["1000"]
    assert navigator.view_structure["VIEW"]["1000"] == ["20", "2000"]
    assert navigator.view_structure["VIEW"]["2000"] == ["79"]
    assert navigator.view_parent_map["VIEW"]["79"] == "2000"


def test_find_paths_direct_inferred_missing_and_unknown_view(navigator):
    assert navigator._find_path_in_view("VIEW", "79") == ("1000", "2000", "79")
    assert navigator._find_path_in_view("VIEW", "89") == ("1000", "2000", "79", "89")
    assert navigator._find_path_in_view("VIEW", "77") is None
    assert navigator._find_path_in_view("MISSING", "79") is None
    assert navigator._build_path("VIEW", "missing") == ("missing",)


def test_get_top_parent_handles_known_roots_unknowns_and_cycles(navigator):
    assert navigator.get_top_parent("89") == "20"
    assert navigator.get_top_parent("20") is None
    assert navigator.get_top_parent("missing") is None

    navigator.weaknesses["1"] = ET.Element("Weakness", ID="1")
    navigator.child_of["1"] = ["2"]
    navigator.child_of["2"] = ["1"]
    assert navigator.get_top_parent("1") is None


def test_lookup_helpers_and_multi_view_paths(navigator):
    paths = navigator.get_paths_in_views("79", ["VIEW", "EMPTY", "MISSING"])

    assert paths == {
        "VIEW": ("1000", "2000", "79"),
        "EMPTY": None,
        "MISSING": None,
    }
    assert navigator.get_element_name("79") == "XSS"
    assert navigator.get_element_name("1000") == "Root Category"
    assert navigator.get_element_name("missing") == "Unknown"
    assert navigator.get_element_type("79") == "Weakness"
    assert navigator.get_element_type("1000") == "Category"
    assert navigator.get_element_type("missing") == "Unknown"


def test_descendants_for_views_categories_weaknesses_and_unknowns(navigator):
    view_desc = navigator.get_descendants("VIEW")
    assert view_desc["type"] == "View"
    assert dict(view_desc["by_view"]["VIEW"]) == {
        1: ["1000"],
        2: ["20", "2000"],
        3: ["79"],
    }

    category_desc = navigator.get_descendants("1000", max_depth=1)
    assert category_desc["type"] == "Category"
    assert dict(category_desc["by_view"]["VIEW"]) == {1: ["20", "2000"]}

    detached_desc = navigator.get_descendants("3000")
    assert detached_desc["type"] == "Category"
    assert dict(detached_desc["by_view"]["no_view"]) == {1: ["999"]}

    assert navigator.get_descendants("79") == {"type": "Weakness", "by_view": {}}
    assert navigator.get_descendants("missing") == {"type": "Unknown", "by_view": {}}


def test_process_category_members_ignores_unknown_categories(navigator):
    before = dict(navigator.view_parent_map["VIEW"])

    navigator._process_category_members("VIEW", "missing")

    assert dict(navigator.view_parent_map["VIEW"]) == before


def test_print_helpers_cover_present_missing_empty_and_top_paths(navigator, capsys):
    navigator.print_paths("79", {"VIEW": ("1000", "2000", "79"), "EMPTY": (), "MISSING": None})
    output = capsys.readouterr().out
    assert "Path found (3 elements)" in output
    assert "Empty path" in output
    assert "CWE not found in this view" in output

    navigator.print_hierarchy("89")
    output = capsys.readouterr().out
    assert "CWE-89: SQL Injection" in output
    assert "CWE-20: Input Validation (TOP)" in output

    navigator.print_hierarchy("20")
    assert "No parent" in capsys.readouterr().out


def test_main_cli_runs_with_hierarchy_and_top_parent(monkeypatch, capsys, cwe_xml):
    from ingestion import cwe_navigator
    argv = ["cwe_navigator", str(cwe_xml), "CWE-89", "--views", "VIEW", "--hierarchy", "--top-parent"]
    monkeypatch.setattr("sys.argv", argv)
    cwe_navigator.main()
    out = capsys.readouterr().out
    assert "ChildOf Hierarchy" in out
    assert "Top Parent" in out
    assert "CWE-89" in out


def test_main_cli_handles_missing_file(monkeypatch, capsys, tmp_path):
    from ingestion import cwe_navigator
    argv = ["cwe_navigator", str(tmp_path / "nope.xml"), "89"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as exc:
        cwe_navigator.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "not found" in err


def test_main_cli_handles_parse_error(monkeypatch, capsys, tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("not really xml <unclosed")
    from ingestion import cwe_navigator
    argv = ["cwe_navigator", str(bad), "89"]
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as exc:
        cwe_navigator.main()
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert "parsing XML file" in err


def test_main_cli_handles_top_parent_missing(monkeypatch, capsys, cwe_xml):
    from ingestion import cwe_navigator
    argv = ["cwe_navigator", str(cwe_xml), "CWE-99999", "--views", "VIEW", "--top-parent"]
    monkeypatch.setattr("sys.argv", argv)
    cwe_navigator.main()
    out = capsys.readouterr().out
    assert "No top parent found" in out

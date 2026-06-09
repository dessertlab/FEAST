import os as _os, sys as _sys
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import xml.etree.ElementTree as ET
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque

class CWENavigator:
    def __init__(self, xml_file_path: str):
        """
        Initializes the navigator with the CWE XML file.

        Args:
            xml_file_path: Path to the cwec_latest.xml file downloaded from MITRE
        """
        self.tree = ET.parse(xml_file_path)
        self.root = self.tree.getroot()

        # Namespace used in the MITRE XML file
        self.ns = {'cwe': 'http://cwe.mitre.org/cwe-7'}

        # Dictionaries to store entities
        self.weaknesses: Dict[str, ET.Element] = {}
        self.categories: Dict[str, ET.Element] = {}
        self.views: Dict[str, ET.Element] = {}

        # Parent relations (ChildOf)
        self.child_of: Dict[str, List[str]] = defaultdict(list)

        # MITRE abstraction level of each weakness: "Pillar"|"Class"|"Base"|"Variant"|"Compound".
        self.abstraction_of: Dict[str, str] = {}

        # Primary ChildOf parent scoped per view: view_id -> {cwe -> parent}.
        # A weakness can sit under several parents and several views; within a view MITRE
        # marks exactly one ChildOf edge as Ordinal="Primary" — the canonical navigation
        # parent for that view. We index it so the canonicalizer can climb a deterministic
        # path (no reliance on XML ordering of secondary edges).
        self.primary_parent_in_view: Dict[str, Dict[str, str]] = defaultdict(dict)

        # View structure: view_id -> {parent_id -> [children_ids]}
        self.view_structure: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

        # Parent->child map for reverse navigation
        self.view_parent_map: Dict[str, Dict[str, str]] = defaultdict(dict)

        self._parse_xml()
        self._build_view_hierarchies()

    def _parse_xml(self):
        """Parses the XML file and builds the basic data structures."""

        # Parse Weaknesses
        for weakness in self.root.findall('.//cwe:Weakness', self.ns):
            cwe_id = weakness.get('ID')
            self.weaknesses[cwe_id] = weakness
            self.abstraction_of[cwe_id] = weakness.get('Abstraction') or ''

            # Extract ChildOf relations, keeping View_ID + Ordinal so we can later climb
            # the *primary* path of a chosen view (e.g. CWE-1000 Research Concepts).
            for rel in weakness.findall('.//cwe:Related_Weakness', self.ns):
                if rel.get('Nature') != 'ChildOf':
                    continue
                parent_id = rel.get('CWE_ID')
                self.child_of[cwe_id].append(parent_id)
                view_id = rel.get('View_ID')
                if view_id and rel.get('Ordinal') == 'Primary':
                    # First primary edge for this (view, cwe) wins; duplicates across the
                    # same view in the XML are ignored to stay deterministic.
                    self.primary_parent_in_view[view_id].setdefault(cwe_id, parent_id)

        # Parse Categories
        for category in self.root.findall('.//cwe:Category', self.ns):
            cat_id = category.get('ID')
            self.categories[cat_id] = category

            # Extract ChildOf relations for categories
            for rel in category.findall('.//cwe:Relationships/cwe:Has_Member', self.ns):
                nature = rel.get('Nature')
                if nature == 'ChildOf':
                    parent_id = rel.get('CWE_ID')
                    self.child_of[cat_id].append(parent_id)

        # Parse Views
        for view in self.root.findall('.//cwe:View', self.ns):
            view_id = view.get('ID')
            self.views[view_id] = view

    def _build_view_hierarchies(self):
        """
        Builds the complete hierarchy of each View.
        """
        for view_id, view in self.views.items():
            # Collect all direct members of the view
            for member in view.findall('.//cwe:Members/cwe:Has_Member', self.ns):
                member_id = member.get('CWE_ID')
                if member_id:
                    # Add to the structure as a child of the view
                    self.view_structure[view_id][view_id].append(member_id)
                    self.view_parent_map[view_id][member_id] = view_id

                    # If it is a category, process recursively
                    if member_id in self.categories:
                        self._process_category_members(view_id, member_id)

    def _process_category_members(self, view_id: str, category_id: str):
        """
        Recursively processes all members of a category.
        """
        if category_id not in self.categories:
            return

        category = self.categories[category_id]

        # Find all members of this category
        for member in category.findall('.//cwe:Relationships/cwe:Has_Member', self.ns):
            member_id = member.get('CWE_ID')

            if not member_id:
                continue

            # Add to the structure
            self.view_structure[view_id][category_id].append(member_id)
            self.view_parent_map[view_id][member_id] = category_id

            # If it is a category, recurse
            if member_id in self.categories:
                self._process_category_members(view_id, member_id)

    def _find_path_in_view(self, view_id: str, cwe_id: str) -> Optional[Tuple[str, ...]]:
        """
        Finds the complete path from the view root up to the specified CWE.
        """
        if view_id not in self.view_parent_map:
            return None

        if cwe_id not in self.view_parent_map[view_id]:
            visited = set()
            current = cwe_id

            while current and current not in visited:
                visited.add(current)
                if current in self.view_parent_map[view_id]:
                    path = self._build_path(view_id, current)
                    if path:
                        if current != cwe_id:
                            return path + (cwe_id,)
                        return path
                    return None

                parents = self.child_of.get(current, [])
                if not parents:
                    break
                current = parents[0]

            return None

        return self._build_path(view_id, cwe_id)

    def _build_path(self, view_id: str, cwe_id: str) -> Optional[Tuple[str, ...]]:
        """
        Builds the path from the view root to the CWE by climbing the parent map.
        """
        path = []
        current = cwe_id
        visited = set()

        while current and current not in visited:
            visited.add(current)
            path.append(current)

            if current == view_id:
                break

            parent = self.view_parent_map[view_id].get(current)
            if not parent:
                break

            current = parent

        if path and path[-1] == view_id:
            path.pop()

        path.reverse()

        return tuple(path) if path else None

    @staticmethod
    def _num(cwe) -> str:
        """Normalise 'CWE-120' / 'cwe-120' / 120 -> '120'."""
        text = str(cwe).strip().upper()
        return text.removeprefix('CWE-') if text.startswith('CWE-') else text

    def abstraction(self, cwe_id) -> Optional[str]:
        """MITRE abstraction level of a weakness, or None if unknown/not a weakness."""
        return self.abstraction_of.get(self._num(cwe_id)) or None

    def primary_parent(self, cwe_id, view: str = "1000") -> Optional[str]:
        """Primary ChildOf parent of ``cwe_id`` within ``view`` (Ordinal=Primary).

        Returns None when the weakness has no primary edge in that view — i.e. it is a
        root/pillar of the view, or it is not placed under that view at all.
        """
        return self.primary_parent_in_view.get(str(view), {}).get(self._num(cwe_id))

    def primary_path(self, cwe_id, view: str = "1000") -> Tuple[str, ...]:
        """Path from ``cwe_id`` up to its view root, following only primary edges.

        Returned leaf-first: ``(cwe, parent, ..., pillar)``. Climbing stops at the first
        node without a primary parent in the view (a pillar/root). Cycle-safe.
        """
        num = self._num(cwe_id)
        path: List[str] = [num]
        seen: Set[str] = {num}
        current = num
        while True:
            parent = self.primary_parent(current, view)
            if not parent or parent in seen:
                break
            path.append(parent)
            seen.add(parent)
            current = parent
        return tuple(path)

    def get_top_parent(self, cwe_id: str) -> Optional[str]:
        """Finds the highest-level parent by climbing all ChildOf relations."""
        if cwe_id not in self.child_of and cwe_id not in self.weaknesses and cwe_id not in self.categories:
            return None

        visited: Set[str] = set()
        current = cwe_id

        while current and current not in visited:
            visited.add(current)
            parents = self.child_of.get(current, [])

            if not parents:
                return current if current != cwe_id else None

            current = parents[0]

        return None

    def get_paths_in_views(self, cwe_id: str, view_ids: List[str]) -> Dict[str, Optional[Tuple[str, ...]]]:
        """Finds the paths of the CWE in multiple views."""
        results = {}

        for view_id in view_ids:
            if view_id not in self.views:
                results[view_id] = None
                continue

            path = self._find_path_in_view(view_id, cwe_id)
            results[view_id] = path

        return results

    def get_element_name(self, element_id: str) -> str:
        """Gets the name of an element (weakness or category)."""
        if element_id in self.weaknesses:
            return self.weaknesses[element_id].get('Name', 'Unknown')
        elif element_id in self.categories:
            return self.categories[element_id].get('Name', 'Unknown')
        else:
            return 'Unknown'

    def get_element_type(self, element_id: str) -> str:
        """Gets the type of an element."""
        if element_id in self.weaknesses:
            return 'Weakness'
        elif element_id in self.categories:
            return 'Category'
        else:
            return 'Unknown'

    def print_paths(self, cwe_id: str, paths: Dict[str, Optional[Tuple[str, ...]]]):
        """Prints the found paths in a readable format."""
        cwe_name = self.get_element_name(cwe_id)
        cwe_type = self.get_element_type(cwe_id)

        print(f"\n{'='*70}")
        print(f"CWE-{cwe_id}: {cwe_name} ({cwe_type})")
        print(f"{'='*70}")

        for view_id, path in paths.items():
            view_name = self.views[view_id].get('Name', 'Unknown') if view_id in self.views else 'Unknown'
            print(f"\nView {view_id}: {view_name}")

            if path is None:
                print(f"   CWE not found in this view")
            elif len(path) == 0:
                print(f"   Empty path")
            else:
                print(f"   Path found ({len(path)} elements):")
                for i, element_id in enumerate(path):
                    element_name = self.get_element_name(element_id)
                    element_type = self.get_element_type(element_id)
                    indent = "   " + "  " * i
                    arrow = "\\->" if i == len(path) - 1 else "+->"
                    highlight = " *" if element_id == cwe_id else ""
                    print(f"{indent}{arrow} CWE-{element_id}: {element_name} ({element_type}){highlight}")

    def print_hierarchy(self, cwe_id: str):
        """Prints the complete ChildOf hierarchy of a CWE."""
        print(f"\n{'='*70}")
        print(f"ChildOf Hierarchy for CWE-{cwe_id}")
        print(f"{'='*70}")

        visited: Set[str] = set()
        current = cwe_id
        path = []

        while current and current not in visited:
            visited.add(current)
            name = self.get_element_name(current)
            type_str = self.get_element_type(current)
            path.append((current, name, type_str))
            parents = self.child_of.get(current, [])
            if not parents:
                break
            current = parents[0]

        if len(path) == 1:
            print("   No parent (already at highest level)")
        else:
            for i, (id_, name, type_str) in enumerate(path):
                indent = "  " * i
                arrow = "\\->" if i == len(path) - 1 else "+->"
                highlight = " (TOP)" if i == len(path) - 1 else ""
                print(f"{indent}{arrow} CWE-{id_}: {name}{highlight}")

    def get_descendants(self, element_id: str, max_depth: Optional[int] = None) -> Dict[str, Dict[int, List[str]]]:
        """Return all descendants of a View, Category, or Weakness."""
        result = {"type": None, "by_view": {}}

        if element_id in self.views:
            result["type"] = "View"
            result["by_view"][element_id] = self._get_descendants_in_view(element_id, element_id, max_depth)
            return result

        if element_id in self.categories:
            result["type"] = "Category"
            for view_id, parents in self.view_parent_map.items():
                if element_id in parents:
                    result["by_view"][view_id] = self._get_descendants_in_view(view_id, element_id, max_depth)
            if not result["by_view"]:
                result["by_view"]["no_view"] = self._get_descendants_in_categories(element_id, max_depth)
            return result

        if element_id in self.weaknesses:
            result["type"] = "Weakness"
            return result

        result["type"] = "Unknown"
        return result

    def _get_descendants_in_view(self, view_id: str, start_id: str, max_depth: Optional[int]) -> Dict[int, List[str]]:
        results = defaultdict(list)
        queue = deque([(start_id, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth != 0:
                results[depth].append(current)
            if max_depth is not None and depth >= max_depth:
                continue
            for child in self.view_structure[view_id].get(current, []):
                queue.append((child, depth + 1))

        return results

    def _get_descendants_in_categories(self, category_id: str, max_depth: Optional[int]) -> Dict[int, List[str]]:
        results = defaultdict(list)
        queue = deque([(category_id, 0)])

        while queue:
            current, depth = queue.popleft()
            if depth != 0:
                results[depth].append(current)
            if max_depth is not None and depth >= max_depth:
                continue
            category = self.categories.get(current)
            if category is None:
                continue
            for member in category.findall('.//cwe:Relationships/cwe:Has_Member', self.ns):
                child_id = member.get('CWE_ID')
                if child_id:
                    queue.append((child_id, depth + 1))

        return results


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='CWE Navigator')
    parser.add_argument('xml_file', help='Path to MITRE cwec_latest.xml file')
    parser.add_argument('cwe_id', help='CWE ID to analyze (e.g., 120)')
    parser.add_argument('--views', nargs='+', default=['700', '888', '1000', '1154'])
    parser.add_argument('--hierarchy', action='store_true')
    parser.add_argument('--top-parent', action='store_true')
    args = parser.parse_args()

    try:
        navigator = CWENavigator(args.xml_file)
        cwe_id = args.cwe_id.removeprefix('CWE-')

        if args.hierarchy:
            navigator.print_hierarchy(cwe_id)
        if args.top_parent:
            top = navigator.get_top_parent(cwe_id)
            if top:
                print(f"\nTop Parent: CWE-{top} - {navigator.get_element_name(top)}")
            else:
                print(f"\nNo top parent found for CWE-{cwe_id}")

        paths = navigator.get_paths_in_views(cwe_id, args.views)
        navigator.print_paths(cwe_id, paths)

    except FileNotFoundError:
        print(f"Error: File '{args.xml_file}' not found", file=sys.stderr)
        sys.exit(1)
    except ET.ParseError as e:
        print(f"Error parsing XML file: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
inspect_root.py

Tiny helper for exploring ROOT files with uproot.

It prints:
- Top-level keys and their class names
- Directory structure (recursively)
- For TTrees: branch names and types
- Optionally: first few entries of a chosen TTree

Usage
-----
Basic structure listing:

    python inspect_root.py myfile.root

Show branches and a few entries from a specific tree:

    python inspect_root.py myfile.root --tree image_tree --show-entries 5

If you're not sure of the tree name, just run without --tree first and
look at the printed keys (TTrees are marked).

With real example file:

    python src/ngimager/tools/inspect_root.py examples/imaging_datasets/NOVO_experiment_DT_at_PTB/autoSorted_coinc_detector_DT-14p8MeV_000041.root

    python src/ngimager/tools/inspect_root.py examples/imaging_datasets/NOVO_experiment_DT_at_PTB/autoSorted_coinc_detector_DT-14p8MeV_000041.root --tree meta

Notes
-----
- Requires `uproot` (`pip install uproot`).
- Works with uproot 4/5-style API.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import uproot


def walk_directory(directory: uproot.reading.ReadOnlyDirectory, prefix: str = "") -> None:
    """
    Recursively print keys and class names under a ROOT directory.
    """
    for key in directory.keys():
        # uproot keys include ";1" etc; keep the short name for readability
        short_name = key.split(";")[0]
        full_path = f"{prefix}/{short_name}" if prefix else short_name
        cls_name = directory.classname_of(key)

        print(f"[{full_path}]  ({cls_name})")

        # Subdirectories are also ReadOnlyDirectory instances
        try:
            obj = directory[short_name]
        except Exception:
            continue

        if isinstance(obj, uproot.reading.ReadOnlyDirectory):
            walk_directory(obj, prefix=full_path)


def describe_tree(tree: uproot.behaviors.TTree.TTree, tree_path: str) -> None:
    """
    Print basic info about a TTree: number of entries and branch info.
    """
    print()
    print(f"=== TTree: {tree_path} ===")
    print(f"Entries: {tree.num_entries}")
    print("Branches:")
    for name, branch in tree.items():
        # branch.interpretation gives type-ish info
        try:
            interp = branch.interpretation
        except Exception:
            interp = "unknown"
        print(f"  - {name}: {interp}")


def show_entries(tree: uproot.behaviors.TTree.TTree, n: int) -> None:
    """
    Print the first n entries. If the tree has exactly one entry,
    pretty-print as key = value lines, which is great for metadata.
    """
    total = tree.num_entries
    n = min(n, total)
    if n <= 0:
        print("\n(No entries requested or available.)")
        return

    arrays = tree.arrays(entry_stop=n, library="np")

    if n == 1:
        print(f"\nSingle entry from tree '{tree.name}':")
        for branch_name, values in arrays.items():
            # values is a 1D array of length 1
            try:
                scalar = values[0]
            except Exception:
                scalar = values
            print(f"  {branch_name} = {scalar!r}")
    else:
        print(f"\nFirst {n} entries from tree '{tree.name}':")
        for branch_name, values in arrays.items():
            print(f"  {branch_name}: {values!r}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore ROOT file structure using uproot."
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to ROOT file",
    )
    parser.add_argument(
        "--tree",
        type=str,
        default=None,
        help="Name or path of a TTree to inspect (e.g. 'image_tree' or 'dir1/dir2/tree')",
    )
    parser.add_argument(
        "--show-entries",
        type=int,
        default=0,
        help="If >0, print the first N entries from the selected tree",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Only print the TTree schema (branches and types) for the selected tree.",
    )
    return parser.parse_args()


def find_tree(
    rootfile: uproot.reading.ReadOnlyDirectory, tree_path: str
) -> Optional[uproot.behaviors.TTree.TTree]:
    """
    Try to resolve a tree path like 'tree', 'dir/tree', etc.
    """
    # Handle simple case first
    if tree_path in rootfile:
        obj = rootfile[tree_path]
        if isinstance(obj, uproot.behaviors.TTree.TTree):
            return obj

    # Try splitting directory-like paths
    parts = tree_path.split("/")
    current = rootfile
    for part in parts:
        if not part:
            continue
        if part not in current:
            return None
        current = current[part]
    if isinstance(current, uproot.behaviors.TTree.TTree):
        return current
    return None


def main() -> None:
    args = parse_args()
    path: Path = args.file

    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    print(f"Opening ROOT file: {path.resolve()}\n")

    with uproot.open(path) as f:
        print("Top-level keys and classes:")
        walk_directory(f)

        if args.tree:
            tree = find_tree(f, args.tree)
            if tree is None:
                print(f"\nCould not find TTree '{args.tree}' in this file.")
                return
            describe_tree(tree, args.tree)
            if args.show_entries > 0:
                show_entries(tree, args.show_entries)
        else:
            print(
                "\nNo --tree specified. If you see a key marked as TTree above,\n"
                "re-run with e.g.:\n\n"
                f"  python {Path(__file__).name} {path.name} --tree <tree_name> --show-entries 5\n"
            )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build lammps_grammar.json from LAMMPS RST documentation.

Scans doc/src/ for ``.. index::`` directives to extract all valid style names
for pair_style, fix, compute, bond_style, angle_style, dihedral_style,
improper_style, and kspace_style. Parses specific RST files to get enum
values for simple commands (units, atom_style).

Usage:
    uv run plugin_lammps/scripts/build_lammps_grammar.py \\
        --lammps-src /path/to/lammps \\
        --output plugin_lammps/data/lammps_grammar.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_COMMANDS_ALL_RST = "Commands_all.rst"

# Pages in Commands_all.rst that are index pages, not actual commands
_NON_COMMAND_REFS = {
    "Commands_all", "Commands_compute", "Commands_dump", "Commands_fix",
    "Commands_kspace", "Commands_pair", "angle", "bond", "dihedral",
    "improper", "kim_commands", "group2ndx", "region2vmd", "neb_spin",
    "fitpod_command",
}


def _index_styles(doc_src: Path, category: str) -> list[str]:
    """Extract all style names for a command category from .. index:: directives."""
    pattern = re.compile(rf"^\.\. index::\s+{re.escape(category)}\s+(.+)", re.M)
    styles: set[str] = set()
    for rst in doc_src.rglob("*.rst"):
        try:
            text = rst.read_text(errors="replace")
        except OSError:
            continue
        for m in pattern.finditer(text):
            name = m.group(1).strip()
            if name:
                styles.add(name)
    return sorted(styles)


def _parse_enum_from_rst(rst_path: Path, keyword: str) -> list[str]:
    """Extract enum values like *lj* or *real* from a command's RST Syntax section."""
    if not rst_path.exists():
        return []
    text = rst_path.read_text(errors="replace")
    # Find lines like: * style = *lj* or *real* or ...
    pattern = re.compile(
        rf"\*\s*{re.escape(keyword)}\*?\s*=\s*((?:\*\S+\*(?:\s+or\s+)?)+)",
        re.I,
    )
    m = pattern.search(text)
    if not m:
        return []
    values = re.findall(r"\*([^*]+)\*", m.group(1))
    return [v.strip() for v in values if v.strip()]


def _top_level_commands(doc_src: Path) -> list[str]:
    """Return top-level command names from Commands_all.rst (authoritative list)."""
    rst_path = doc_src / _COMMANDS_ALL_RST
    if not rst_path.exists():
        return []
    text = rst_path.read_text(errors="replace")
    # Extract the link target names: :doc:`label <name>` → name
    names = re.findall(r"<(\w+)>`", text)
    commands = {n for n in names if n not in _NON_COMMAND_REFS}
    return sorted(commands)


def build(lammps_src: Path) -> dict:
    doc_src = lammps_src / "doc" / "src"
    if not doc_src.exists():
        print(f"ERROR: doc/src not found under {lammps_src}", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {doc_src} ...")

    grammar: dict = {}

    # Simple enum commands — parse their specific RST files
    grammar["units_styles"] = _parse_enum_from_rst(doc_src / "units.rst", "style") or [
        "lj", "real", "metal", "si", "cgs", "electron", "micro", "nano"
    ]

    raw_atom = _parse_enum_from_rst(doc_src / "atom_style.rst", "style")
    grammar["atom_styles"] = raw_atom or [
        "atomic", "full", "molecular", "charge", "bond", "angle", "sphere",
        "ellipsoid", "line", "tri", "body", "dipole", "electron", "hybrid",
        "amoeba", "dpd", "edpd", "mdpd", "smd", "sph", "spin", "wavepacket",
    ]

    # Index-based style extraction
    for category, key in [
        ("pair_style",      "pair_styles"),
        ("fix",             "fix_styles"),
        ("compute",         "compute_styles"),
        ("bond_style",      "bond_styles"),
        ("angle_style",     "angle_styles"),
        ("dihedral_style",  "dihedral_styles"),
        ("improper_style",  "improper_styles"),
        ("kspace_style",    "kspace_styles"),
    ]:
        styles = _index_styles(doc_src, category)
        grammar[key] = styles
        print(f"  {key}: {len(styles)} styles")

    grammar["top_level_commands"] = _top_level_commands(doc_src)
    print(f"  top_level_commands: {len(grammar['top_level_commands'])}")

    return grammar


def main() -> None:
    parser = argparse.ArgumentParser(description="Build lammps_grammar.json from RST docs")
    parser.add_argument("--lammps-src", required=True, type=Path,
                        help="Path to LAMMPS source tree (must contain doc/src/)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output path for lammps_grammar.json")
    args = parser.parse_args()

    grammar = build(args.lammps_src.resolve())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(grammar, indent=2) + "\n")
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()

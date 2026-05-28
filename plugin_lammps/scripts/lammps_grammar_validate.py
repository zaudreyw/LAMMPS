"""Grammar-based validation for LAMMPS .in scripts.

Analogous to ``xmllint --schema`` for GEOS XML: checks that every command
and style name in a .in script is a known LAMMPS command/style, using a
grammar extracted from the LAMMPS RST documentation.

Public API
----------
load_grammar(path=None) -> dict
    Load lammps_grammar.json.  Falls back to a minimal built-in grammar if
    the file is not found.

grammar_errors(path_or_text, grammar) -> list[str]
    Return a list of human-readable error strings.  Empty list means the
    script passes grammar validation.
"""
from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path
from typing import Union

# ---------------------------------------------------------------------------
# Grammar file location
# ---------------------------------------------------------------------------

_DEFAULT_GRAMMAR_PATH = Path(
    os.environ.get(
        "LAMMPS_GRAMMAR_PATH",
        str(Path(__file__).resolve().parent.parent / "data" / "lammps_grammar.json"),
    )
)

# ---------------------------------------------------------------------------
# Minimal fallback grammar (used when grammar.json is not available)
# ---------------------------------------------------------------------------

_FALLBACK_GRAMMAR: dict[str, list[str]] = {
    "units_styles": ["lj", "real", "metal", "si", "cgs", "electron", "micro", "nano"],
    "atom_styles": [
        "atomic", "full", "molecular", "charge", "bond", "angle", "sphere",
        "ellipsoid", "line", "tri", "body", "dipole", "electron", "hybrid",
    ],
    "pair_styles": [
        "lj/cut", "lj/cut/coul/cut", "lj/cut/coul/long", "lj/smooth",
        "lj/smooth/linear", "lj/charmm/coul/charmm", "lj/charmm/coul/long",
        "eam", "eam/alloy", "eam/fs", "buck", "morse", "sw", "tersoff",
        "tersoff/zbl", "reaxff", "table", "zero", "none", "hybrid",
        "hybrid/overlay", "airebo", "comb", "comb3",
    ],
    "fix_styles": [
        "nve", "nvt", "npt", "nph", "langevin", "temp/berendsen",
        "temp/rescale", "nvt/sllod", "nve/sphere", "nvt/sphere", "npt/sphere",
        "deform", "indent", "spring", "gravity", "freeze", "momentum",
        "recenter", "wall/reflect", "wall/lj93", "wall/lj126", "viscous",
        "rigid", "rigid/nve", "rigid/nvt", "rigid/npt", "store/state",
        "store/force", "enforce2d", "deposit", "evaporate", "heat",
        "thermal/conductivity", "viscosity", "move", "shake", "rattle",
    ],
    "compute_styles": [
        "temp", "temp/partial", "temp/deform", "temp/com", "pressure",
        "pe", "ke", "msd", "rdf", "vacf", "com", "gyration", "bond",
        "angle", "stress/atom", "displace/atom", "reduce",
    ],
    "bond_styles": [
        "harmonic", "fene", "fene/expand", "morse", "table", "none",
        "zero", "hybrid", "class2",
    ],
    "angle_styles": [
        "harmonic", "cosine", "charmm", "table", "none", "zero", "hybrid",
        "class2",
    ],
    "dihedral_styles": [
        "harmonic", "charmm", "opls", "table", "none", "zero", "hybrid",
        "class2",
    ],
    "improper_styles": ["harmonic", "none", "zero", "hybrid", "class2"],
    "kspace_styles": [
        "pppm", "pppm/cg", "pppm/tip4p", "pppm/disp", "ewald", "msm",
        "msm/cg",
    ],
    "top_level_commands": [],  # empty fallback: unknown-command check disabled
}

# ---------------------------------------------------------------------------
# Commands whose first argument is the style name (token index 1)
# ---------------------------------------------------------------------------
_STYLE_AT_1 = {
    "units":          "units_styles",
    "atom_style":     "atom_styles",
    "pair_style":     "pair_styles",
    "bond_style":     "bond_styles",
    "angle_style":    "angle_styles",
    "dihedral_style": "dihedral_styles",
    "improper_style": "improper_styles",
    "kspace_style":   "kspace_styles",
}

# Commands whose style name is at token index 3 (fix/compute: ID group style ...)
_STYLE_AT_3 = {
    "fix":     "fix_styles",
    "compute": "compute_styles",
}

# Commands that are always valid with any arguments (variable, shell, etc.)
_UNCHECKED_ARGS = {
    "variable", "shell", "python", "include", "jump", "label", "if",
    "next", "print", "echo", "log", "info", "timer", "plugin", "suffix",
    "package", "partition",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_grammar(path: Union[str, Path, None] = None) -> dict[str, set[str]]:
    """Load grammar from JSON file, falling back to built-in minimal grammar."""
    p = Path(path) if path else _DEFAULT_GRAMMAR_PATH
    if p.exists():
        try:
            raw = json.loads(p.read_text())
            return {k: set(v) for k, v in raw.items() if isinstance(v, list)}
        except (json.JSONDecodeError, OSError):
            pass
    return {k: set(v) for k, v in _FALLBACK_GRAMMAR.items()}


def grammar_errors(path_or_text: Union[str, Path], grammar: dict[str, set[str]]) -> list[str]:
    """Validate a LAMMPS .in script against the grammar.

    Parameters
    ----------
    path_or_text:
        Either a Path to a .in file, or the raw script text as a string.
    grammar:
        Grammar dict from :func:`load_grammar`.

    Returns
    -------
    List of error strings. Empty list means the script passes.
    """
    p = Path(path_or_text)
    if p.exists():
        try:
            text = p.read_text(errors="replace")
        except OSError as exc:
            return [f"Could not read file: {exc}"]
    else:
        text = str(path_or_text)

    errors: list[str] = []
    top_level = grammar.get("top_level_commands", set())
    continuation = False  # True if previous line ended with &

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#")[0].strip()
        if not line:
            continuation = False
            continue
        # LAMMPS line continuation: line ending with & means next line is part
        # of the same command and should not be parsed as a new command.
        if continuation:
            continuation = line.endswith("&")
            continue
        continuation = line.endswith("&")
        tokens = line.rstrip("&").split()
        if not tokens:
            continue
        cmd = tokens[0].lower()

        # Skip commands whose arguments are free-form
        if cmd in _UNCHECKED_ARGS:
            continue

        # Check that the command is a known top-level command
        if top_level and cmd not in top_level:
            suggestion = _suggest(cmd, top_level)
            errors.append(
                f"Line {lineno}: unknown command '{cmd}'"
                + (f" — did you mean '{suggestion}'?" if suggestion else "")
            )
            continue  # no point checking style if command is unknown

        # Check style argument at position 1 (units, pair_style, atom_style, ...)
        if cmd in _STYLE_AT_1:
            grammar_key = _STYLE_AT_1[cmd]
            valid = grammar.get(grammar_key, set())
            if valid and len(tokens) >= 2:
                style = tokens[1].lower()
                if style not in valid:
                    suggestion = _suggest(style, valid)
                    errors.append(
                        f"Line {lineno}: unknown {cmd} style '{tokens[1]}'"
                        + (f" — did you mean '{suggestion}'?" if suggestion else "")
                        + f" (valid: {', '.join(sorted(list(valid))[:8])}{'...' if len(valid) > 8 else ''})"
                    )

        # Check style argument at position 3 (fix/compute: ID group style ...)
        elif cmd in _STYLE_AT_3:
            grammar_key = _STYLE_AT_3[cmd]
            valid = grammar.get(grammar_key, set())
            if valid and len(tokens) >= 4:
                style = tokens[3].lower()
                if style not in valid:
                    suggestion = _suggest(style, valid)
                    errors.append(
                        f"Line {lineno}: unknown {cmd} style '{tokens[3]}'"
                        + (f" — did you mean '{suggestion}'?" if suggestion else "")
                    )

    return errors


def _suggest(name: str, valid: set[str]) -> str | None:
    """Return the closest match from valid styles, or None."""
    matches = difflib.get_close_matches(name.lower(), [v.lower() for v in valid], n=1, cutoff=0.6)
    if not matches:
        return None
    # Return the original-case version
    target = matches[0]
    for v in valid:
        if v.lower() == target:
            return v
    return target


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Validate a LAMMPS .in script against the grammar")
    parser.add_argument("in_file", type=Path, help="Path to .in script")
    parser.add_argument("--grammar", type=Path, default=None,
                        help="Path to lammps_grammar.json (default: auto-locate)")
    args = parser.parse_args()

    g = load_grammar(args.grammar)
    errs = grammar_errors(args.in_file, g)
    if errs:
        for e in errs:
            print(e)
        sys.exit(1)
    else:
        print(f"OK: {args.in_file}")
        sys.exit(0)

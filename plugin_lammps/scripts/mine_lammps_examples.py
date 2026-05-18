#!/usr/bin/env python3
"""Mine LAMMPS example scripts to generate eval task skeletons.

Scans a LAMMPS source tree, finds all example .in scripts, parses key
parameters, and produces:

  1. lammps_example_pairs.jsonl — task_id → source .in path mapping
     (one line per .in file found under examples/).
  2. Optionally, per-task instructions.txt templates under --output-dir.

The mining logic is intentionally different from GEOS:
  - GEOS tasks were mined from RST tutorial pages (task spec from prose docs).
  - LAMMPS tasks are mined directly from .in scripts (task spec auto-generated
    from parsed script metadata + optional README context).

Key differences from GEOS mining (per may17update report):
  - No single authoritative "task spec" document (LAMMPS examples have minimal
    README files, not structured tutorial prose).
  - No variant suffix expansion needed (LAMMPS .in files have no _base/_smoke
    naming convention).
  - No RST cross-reference blocking (LAMMPS blocks source .in file directly).

Usage:
    uv run plugin_lammps/scripts/mine_lammps_examples.py \\
        --lammps-src /path/to/lammps \\
        --pairs-file data/eval/lammps_example_pairs.jsonl \\
        --output-dir data/eval/experiments_lammps_mined \\
        --dry-run

    # Only update the pairs file, no instructions.txt generation:
    uv run plugin_lammps/scripts/mine_lammps_examples.py \\
        --lammps-src /path/to/lammps \\
        --pairs-file data/eval/lammps_example_pairs.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import textwrap
from pathlib import Path


# ---------------------------------------------------------------------------
# Metadata extraction (mirrors build_lammps_vector_db.py._extract_in_metadata)
# ---------------------------------------------------------------------------

def _extract_in_metadata(text: str) -> dict[str, str]:
    """Parse a LAMMPS .in script and return key command values."""
    meta: dict[str, str] = {}
    pair_styles: list[str] = []
    fix_styles: list[str] = []
    computes: list[str] = []
    for line in text.splitlines():
        stripped = line.split("#")[0].strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if not tokens:
            continue
        cmd = tokens[0].lower()
        rest = " ".join(tokens[1:])
        if cmd == "units" and "units" not in meta:
            meta["units"] = rest.split()[0] if rest else ""
        elif cmd == "atom_style" and "atom_style" not in meta:
            meta["atom_style"] = rest.split()[0] if rest else ""
        elif cmd == "dimension" and "dimension" not in meta:
            meta["dimension"] = rest.split()[0] if rest else "3"
        elif cmd == "pair_style":
            pair_styles.append(rest.split()[0] if rest else rest)
        elif cmd == "bond_style" and "bond_style" not in meta:
            meta["bond_style"] = rest.split()[0] if rest else ""
        elif cmd == "fix" and len(tokens) >= 4:
            fix_styles.append(tokens[3])
        elif cmd == "compute" and len(tokens) >= 4:
            computes.append(tokens[3])
        elif cmd == "run" and "run" not in meta:
            meta["run"] = rest.split()[0] if rest else ""
        elif cmd == "minimize" and "run" not in meta:
            meta["run"] = "minimize"
        elif cmd == "read_data" and "read_data" not in meta:
            meta["read_data"] = rest.split()[0] if rest else ""
        elif cmd == "lattice" and "lattice" not in meta:
            meta["lattice"] = rest.split()[0] if rest else ""
    meta["pair_styles"] = ", ".join(dict.fromkeys(pair_styles)) if pair_styles else ""
    meta["fix_styles"] = ", ".join(dict.fromkeys(fix_styles)) if fix_styles else ""
    meta["computes"] = ", ".join(dict.fromkeys(computes)) if computes else ""
    return meta


def _read_readme(example_dir: Path) -> str:
    """Return README content from an example directory, if any."""
    for name in ("README", "README.md", "readme", "readme.txt"):
        p = example_dir / name
        if p.exists():
            try:
                return p.read_text(errors="replace")[:2000]
            except OSError:
                pass
    return ""


def _make_task_id(in_path: Path, lammps_src: Path) -> str:
    """Derive a snake_case task_id from the example directory + filename."""
    try:
        rel = in_path.relative_to(lammps_src / "examples")
    except ValueError:
        rel = in_path.relative_to(lammps_src)
    parts = list(rel.parts)
    # e.g. ['melt', 'in.melt'] → 'melt'
    # e.g. ['flow', 'in.flow.2d'] → 'flow_2d'
    dir_part = parts[0] if len(parts) > 1 else ""
    file_stem = rel.name
    # Strip leading 'in.' or trailing '.in'
    if file_stem.startswith("in."):
        file_stem = file_stem[3:]
    elif file_stem.endswith(".in"):
        file_stem = file_stem[:-3]
    # Normalize dots/dashes to underscores
    file_stem = re.sub(r"[.\-/]", "_", file_stem)
    file_stem = re.sub(r"_+", "_", file_stem).strip("_")
    dir_part = re.sub(r"[.\-/]", "_", dir_part).strip("_")

    if dir_part and not file_stem.startswith(dir_part):
        task_id = f"{dir_part}_{file_stem}" if file_stem else dir_part
    else:
        task_id = file_stem or dir_part
    return task_id.lower()


def _generate_instructions(
    in_path: Path,
    text: str,
    meta: dict[str, str],
    readme: str,
    task_id: str,
    source_relpath: str,
) -> str:
    """Auto-generate an instructions.txt template from script metadata."""
    units = meta.get("units", "lj")
    atom_style = meta.get("atom_style", "atomic")
    dim = meta.get("dimension", "3")
    lattice = meta.get("lattice", "")
    pair = meta.get("pair_styles", "")
    fixes = meta.get("fix_styles", "")
    computes = meta.get("computes", "")
    run_val = meta.get("run", "")
    has_data = bool(meta.get("read_data", ""))
    bond_style = meta.get("bond_style", "")

    lines: list[str] = []
    # Headline — infer physics type from metadata
    if "nvt" in fixes or "npt" in fixes:
        ensemble = "NVT" if "nvt" in fixes else "NPT"
    elif "nve" in fixes:
        ensemble = "NVE"
    elif run_val == "minimize":
        ensemble = "energy minimization"
    else:
        ensemble = "NVE/NVT"

    physics_hint = ""
    if "sllod" in fixes:
        physics_hint = "NEMD Couette flow (SLLOD algorithm)"
    elif "deform" in fixes:
        physics_hint = "deformation simulation"
    elif "indent" in fixes:
        physics_hint = "nanoindentation"
    elif "msd" in computes:
        physics_hint = "diffusion via MSD calculation"
    elif "shake" in fixes or "rattle" in fixes:
        physics_hint = "constrained molecular dynamics"
    elif dim == "2":
        physics_hint = "2D simulation"
    elif "fcc" in lattice:
        physics_hint = "FCC crystal simulation"
    else:
        physics_hint = "molecular dynamics simulation"

    lines.append(f"Write a LAMMPS input script for a {physics_hint}.")
    lines.append(f"This task is derived from the LAMMPS example: {source_relpath}")
    lines.append("")

    lines.append("System parameters:")
    lines.append(f"- {units} units")
    lines.append(f"- Atom style: {atom_style}")
    if dim != "3":
        lines.append(f"- {dim}D simulation")
    if lattice:
        lines.append(f"- Lattice: {lattice}")
    if pair:
        lines.append(f"- Pair style: {pair}")
    if bond_style:
        lines.append(f"- Bond style: {bond_style}")
    if has_data:
        lines.append("- Reads initial geometry from a data file")
    lines.append("")

    if readme:
        lines.append("Context from LAMMPS example README:")
        for readme_line in readme.splitlines()[:10]:
            lines.append(f"  {readme_line}")
        lines.append("")

    lines.append("Procedure:")
    lines.append("1. Set up the system geometry.")
    if run_val == "minimize":
        lines.append("2. Minimize energy to relax the structure.")
    else:
        lines.append(f"2. Run {ensemble} dynamics.")
        if run_val:
            lines.append(f"3. Run for {run_val} timesteps.")
    lines.append("")
    lines.append("Write all output files to /workspace/inputs/.")
    lines.append(f"[NOTE: Auto-generated from {source_relpath}. Review and edit before use.]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _in_files(examples_dir: Path) -> list[Path]:
    results: list[Path] = []
    for p in examples_dir.rglob("*"):
        if p.is_file():
            name = p.name
            if name.startswith("in.") or name.endswith(".in"):
                results.append(p)
    return sorted(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine LAMMPS example scripts into eval task skeletons"
    )
    parser.add_argument("--lammps-src", required=True, type=Path,
                        help="Path to LAMMPS source tree (must contain examples/)")
    parser.add_argument("--pairs-file", required=True, type=Path,
                        help="Output lammps_example_pairs.jsonl path")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="If set, write instructions.txt per task under this directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be written without writing")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N examples (for testing)")
    args = parser.parse_args()

    lammps_src = args.lammps_src.resolve()
    examples_dir = lammps_src / "examples"
    if not examples_dir.exists():
        print(f"ERROR: examples/ not found under {lammps_src}", file=sys.stderr)
        sys.exit(1)

    in_paths = _in_files(examples_dir)
    if args.limit:
        in_paths = in_paths[: args.limit]

    print(f"Found {len(in_paths)} .in files under {examples_dir}")

    pairs: list[dict] = []
    for in_path in in_paths:
        try:
            text = in_path.read_text(errors="replace")
        except OSError as exc:
            print(f"  WARN: skip {in_path}: {exc}")
            continue

        try:
            source_relpath = str(in_path.relative_to(lammps_src))
        except ValueError:
            source_relpath = str(in_path)

        task_id = _make_task_id(in_path, lammps_src)
        meta = _extract_in_metadata(text)
        readme = _read_readme(in_path.parent)

        pair = {
            "task_id": task_id,
            "lammps_example_relpaths": [source_relpath],
        }
        pairs.append(pair)

        if args.output_dir is not None:
            instructions = _generate_instructions(
                in_path, text, meta, readme, task_id, source_relpath
            )
            task_dir = args.output_dir / task_id
            if not args.dry_run:
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / "instructions.txt").write_text(instructions)
            else:
                print(f"  [DRY] Would write {task_dir / 'instructions.txt'}")
                print(textwrap.indent(instructions[:300], "    "))
                print("    ...")

    if args.dry_run:
        print(f"\n[DRY] Would write {len(pairs)} entries to {args.pairs_file}")
        for p in pairs[:5]:
            print(f"  {json.dumps(p)}")
        if len(pairs) > 5:
            print(f"  ... ({len(pairs) - 5} more)")
    else:
        args.pairs_file.parent.mkdir(parents=True, exist_ok=True)
        with args.pairs_file.open("w") as fh:
            for pair in pairs:
                fh.write(json.dumps(pair) + "\n")
        print(f"\nWrote {len(pairs)} entries to {args.pairs_file}")
        if args.output_dir:
            print(f"Wrote instructions.txt files under {args.output_dir}")

    # Summary statistics
    unique_dirs = {Path(p["lammps_example_relpaths"][0]).parent.name for p in pairs}
    print(f"\nMining summary:")
    print(f"  Total .in files found   : {len(pairs)}")
    print(f"  Unique example dirs     : {len(unique_dirs)}")
    print(f"  Mining method           : direct .in script scan (no RST/tutorial parsing)")
    print(f"  vs GEOS                 : GEOS mines from RST tutorial prose; LAMMPS mines")
    print(f"                            directly from .in scripts (no structured spec doc)")


if __name__ == "__main__":
    main()

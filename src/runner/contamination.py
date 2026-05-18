"""Contamination prevention for GEOS agent evaluation.

For a given task, the agent must not have access to:

1. The task's ground-truth XML files (they are the answer).
2. Variant siblings of those files in the GEOS source tree.  Given a GT
   basename ``Foo_base.xml``, the source often also contains
   ``Foo_benchmark.xml``, ``Foo_smoke.xml``, ``Foo_base_iterative.xml`` —
   these share almost all parameters and would leak the answer.
3. The tutorial/example RST that the task was mined from, as listed in
   ``example_pairs.jsonl`` (maps ``task_id`` → RST path relative to the
   GEOS source root).

This module provides two things:

- :func:`get_blocked_files_for_task` — returns the block list (XML
  basenames + RST relative paths) for one task.
- :func:`create_filtered_geos_copy` — hardlinks the GEOS source tree into
  a throwaway directory, omitting blocked files.  Hardlinks are safer than
  symlinks for a Docker bind-mount because they can't be followed to a
  blocked target: the file simply isn't in the mount.

Design notes
------------
The block-list logic is ported from ``geos_agent/scripts/eval/contamination.py``
(which had variant expansion but used a symlink tree).  The sanitized-copy
mechanism is the hardlink approach originally in ``run_eval.py``.  Combining
them closes two gaps that existed in repo3's runner: variant expansion and
RST blocking.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_GEOS_SOURCE_DIR = Path(
    os.environ.get(
        "GEOS_SOURCE_DIR",
        "/data/shared/geophysics_agent_data/data/GEOS",
    )
)

_DEFAULT_EXAMPLE_PAIRS = Path(
    os.environ.get(
        "GEOS_EXAMPLE_PAIRS",
        "/data/shared/geophysics_agent_data/data/eval/example_pairs.jsonl",
    )
)


# ---------------------------------------------------------------------------
# Variant stem handling (mirrors logic in geos_agent context.py)
# ---------------------------------------------------------------------------

_XML_VARIANT_SUFFIXES = (
    "_base_iterative",
    "_base_direct",
    "_iterative_base",
    "_direct_base",
    "_iterative",
    "_direct",
    "_benchmark",
    "_smoke",
    "_base",
)

_GENERIC_XML_STEMS = {
    "base", "benchmark", "input", "inputs", "problem", "model", "smoke",
}

_EXAMPLE_LABEL_RE = re.compile(r"\s*\.\.\s*_([^:]+):")


def _xml_stem_keys(filename: str) -> set[str]:
    """Normalized stem keys for an XML filename.

    Strips known variant suffixes so that ``Foo_base.xml``,
    ``Foo_benchmark.xml``, ``Foo_smoke.xml``, and ``Foo_base_iterative.xml``
    all reduce to ``{"foo"}``.

    Keys shorter than 10 chars or in :data:`_GENERIC_XML_STEMS` are dropped,
    so generic filenames like ``base.xml`` don't collide across unrelated
    examples.
    """
    stem = Path(filename).stem.lower()
    keys: set[str] = set()
    pending = [stem]
    while pending:
        s = pending.pop()
        if s in keys:
            continue
        keys.add(s)
        for suffix in _XML_VARIANT_SUFFIXES:
            if s.endswith(suffix):
                stripped = s[: -len(suffix)]
                if stripped and stripped not in keys:
                    pending.append(stripped)
    return {k for k in keys if len(k) >= 10 and k not in _GENERIC_XML_STEMS}


def _expand_blocked_xml_with_variants(
    blocked_basenames: Iterable[str],
    geos_source_dir: Path,
) -> set[str]:
    """Given GT basenames, also block variant siblings in the GEOS source."""
    blocked_basenames = [b.lower() for b in blocked_basenames]
    blocked_keys: set[str] = set()
    for name in blocked_basenames:
        blocked_keys.update(_xml_stem_keys(name))

    if not blocked_keys:
        return set(blocked_basenames)

    expanded: set[str] = set(blocked_basenames)
    for xml_path in geos_source_dir.rglob("*.xml"):
        if not xml_path.is_file():
            continue
        if _xml_stem_keys(xml_path.name) & blocked_keys:
            expanded.add(xml_path.name.lower())
    return expanded


# ---------------------------------------------------------------------------
# RST mapping
# ---------------------------------------------------------------------------

def _load_example_rst_mappings(
    example_pairs_path: Path | None = None,
) -> dict[str, str]:
    """Load ``task_id`` → RST path (relative to GEOS source root) from
    ``example_pairs.jsonl``.  Each line: JSON object with ``title`` (RST
    directive like ``.. _buckleyLeverettProblem:``) and ``rst_path``.
    """
    path = Path(example_pairs_path or _DEFAULT_EXAMPLE_PAIRS)
    if not path.exists():
        logger.warning("example_pairs.jsonl not found at %s", path)
        return {}

    mappings: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            match = _EXAMPLE_LABEL_RE.match(str(record.get("title", "")))
            rst_path = str(record.get("rst_path", "")).strip()
            if match and rst_path:
                mappings[match.group(1)] = rst_path
    return mappings


# ---------------------------------------------------------------------------
# Collecting GT files
# ---------------------------------------------------------------------------

def _collect_gt_xml_basenames(gt_experiment_dir: Path) -> list[str]:
    """Lowercased XML basenames under one ground-truth experiment directory."""
    if not gt_experiment_dir.exists():
        return []
    return sorted(
        {p.name.lower() for p in gt_experiment_dir.rglob("*.xml") if p.is_file()}
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_blocked_files_for_task(
    task_id: str,
    ground_truth_dir: str | Path,
    *,
    geos_source_dir: str | Path | None = None,
    example_pairs_path: str | Path | None = None,
    expand_variants: bool = True,
) -> dict[str, list[str]]:
    """Return files to hide from the agent for *task_id*.

    Returns
    -------
    dict with:

    - ``blocked_xml_filenames`` — lowercased XML basenames from the GT dir,
      optionally expanded with variant siblings found in the GEOS source.
    - ``blocked_rst_paths`` — RST paths (relative to the GEOS source root)
      from ``example_pairs.jsonl`` that must not be accessible.
    """
    gt_dir = Path(ground_truth_dir) / task_id
    blocked_exact = _collect_gt_xml_basenames(gt_dir)

    if expand_variants:
        src_root = Path(geos_source_dir or _DEFAULT_GEOS_SOURCE_DIR)
        if src_root.exists():
            blocked_xml = sorted(
                _expand_blocked_xml_with_variants(blocked_exact, src_root)
            )
        else:
            logger.warning(
                "GEOS source dir %s not found; skipping variant expansion", src_root
            )
            blocked_xml = blocked_exact
    else:
        blocked_xml = blocked_exact

    rst_mappings = _load_example_rst_mappings(
        Path(example_pairs_path) if example_pairs_path else None
    )
    blocked_rst = [rst_mappings[task_id]] if task_id in rst_mappings else []

    return {
        "blocked_xml_filenames": blocked_xml,
        "blocked_rst_paths": blocked_rst,
    }


def create_filtered_geos_copy(
    geos_src: Path,
    *,
    blocked_xml_basenames: Iterable[str],
    blocked_rst_relpaths: Iterable[str] = (),
    tmp_parent: Path,
) -> Path:
    """Hardlink-copy *geos_src* into a fresh temp dir, omitting blocked files.

    Parameters
    ----------
    geos_src:
        Source GEOS tree to mirror.
    blocked_xml_basenames:
        Lowercased XML basenames to exclude (e.g. ``'deadoil_base.xml'``).
    blocked_rst_relpaths:
        GEOS-relative RST paths to exclude
        (e.g. ``'src/docs/sphinx/basicExamples/multiphaseFlow/Example.rst'``).
    tmp_parent:
        Directory under which the unique temp copy is created. Must be on
        the same filesystem as *geos_src* for hardlinks to succeed (falls
        back to a real copy if ``os.link`` fails).

    Returns
    -------
    Path to the sanitized GEOS root (``<tmp_parent>/geos_eval_<rand>/geos``).
    Pass this root to :func:`cleanup_filtered_geos_copy` when done.
    """
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=tmp_parent, prefix="geos_eval_"))
    geos_dest = tmp_dir / "geos"

    blocked_xml_lower = {n.lower() for n in blocked_xml_basenames}
    blocked_rst_lower = {
        p.replace("\\", "/").lower() for p in blocked_rst_relpaths if p
    }

    def _ignore(src_dir: str, names: list[str]) -> set[str]:
        skipped: set[str] = set()
        for name in names:
            if name.lower() in blocked_xml_lower:
                skipped.add(name)
                continue
            if blocked_rst_lower:
                try:
                    rel = (Path(src_dir) / name).relative_to(geos_src)
                    if str(rel).replace("\\", "/").lower() in blocked_rst_lower:
                        skipped.add(name)
                except ValueError:
                    pass
        return skipped

    def _hardlink_or_copy(src: str, dst: str) -> None:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    shutil.copytree(
        geos_src, geos_dest,
        ignore=_ignore,
        copy_function=_hardlink_or_copy,
        symlinks=True,
    )
    return geos_dest


def cleanup_filtered_geos_copy(geos_copy: Path) -> None:
    """Remove the temp directory created by :func:`create_filtered_geos_copy`."""
    shutil.rmtree(geos_copy.parent, ignore_errors=True)


# ---------------------------------------------------------------------------
# LAMMPS contamination
#
# Design note — RAG vs. Agentic File Navigation:
#   The LAMMPS RAG server already excludes GT-corresponding filenames from
#   search results via EXCLUDED_GT_IN_FILENAMES. This module provides the
#   complementary filesystem-level decontamination: a hardlink copy of the
#   LAMMPS source tree with the exact source .in files removed, so the agent
#   cannot read them via Bash/Read either. Together these two mechanisms ensure
#   that the RAG path and the file-navigation path are both decontaminated.
# ---------------------------------------------------------------------------

_DEFAULT_LAMMPS_EXAMPLE_PAIRS = Path(
    __import__("os").environ.get(
        "LAMMPS_EXAMPLE_PAIRS",
        str(Path(__file__).resolve().parents[2] / "data" / "eval" / "lammps_example_pairs.jsonl"),
    )
)


def _load_lammps_example_pairs(pairs_path: Path | None = None) -> dict[str, list[str]]:
    """Load ``task_id`` → list of LAMMPS source .in relpaths from JSONL file."""
    path = Path(pairs_path or _DEFAULT_LAMMPS_EXAMPLE_PAIRS)
    if not path.exists():
        logger.warning("lammps_example_pairs.jsonl not found at %s", path)
        return {}
    mappings: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_id = str(record.get("task_id", "")).strip()
            relpaths = record.get("lammps_example_relpaths", [])
            if task_id and relpaths:
                mappings[task_id] = [str(p) for p in relpaths]
    return mappings


def get_blocked_in_files_for_task(
    task_id: str,
    ground_truth_dir: str | Path,
    *,
    lammps_example_pairs_path: str | Path | None = None,
) -> dict[str, list[str]]:
    """Return .in files to hide from the agent for *task_id* in LAMMPS mode.

    Returns
    -------
    dict with:

    - ``blocked_in_basenames`` — lowercased .in basenames from the GT dir
      (passed to the RAG server via EXCLUDED_GT_IN_FILENAMES).
    - ``blocked_in_source_relpaths`` — paths relative to the LAMMPS source root
      to remove from the filesystem copy (from lammps_example_pairs.jsonl).
    """
    gt_dir = Path(ground_truth_dir) / task_id
    blocked_basenames: list[str] = []
    if gt_dir.exists():
        blocked_basenames = sorted({
            p.name.lower()
            for p in gt_dir.rglob("*")
            if p.is_file() and (p.name.lower().endswith(".in") or p.name.lower().startswith("in."))
        })

    pairs = _load_lammps_example_pairs(
        Path(lammps_example_pairs_path) if lammps_example_pairs_path else None
    )
    source_relpaths = pairs.get(task_id, [])

    return {
        "blocked_in_basenames": blocked_basenames,
        "blocked_in_source_relpaths": source_relpaths,
    }


def create_filtered_lammps_copy(
    lammps_src: Path,
    *,
    blocked_in_source_relpaths: Iterable[str] = (),
    tmp_parent: Path,
) -> Path:
    """Hardlink-copy *lammps_src* into a fresh temp dir, omitting blocked .in files.

    Parameters
    ----------
    lammps_src:
        LAMMPS source tree to mirror.
    blocked_in_source_relpaths:
        Paths relative to *lammps_src* to exclude
        (e.g. ``'examples/melt/in.melt'``).
    tmp_parent:
        Directory under which the unique temp copy is created. Should be on
        the same filesystem as *lammps_src* for efficient hardlinks.

    Returns
    -------
    Path to the sanitized LAMMPS root at
    ``<tmp_parent>/lammps_eval_<rand>/lammps``.
    Pass to :func:`cleanup_filtered_lammps_copy` when done.
    """
    tmp_parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=tmp_parent, prefix="lammps_eval_"))
    lammps_dest = tmp_dir / "lammps"

    blocked_norm = {
        str(p).replace("\\", "/").lower()
        for p in blocked_in_source_relpaths
        if p
    }

    def _ignore(src_dir: str, names: list[str]) -> set[str]:
        skipped: set[str] = set()
        if not blocked_norm:
            return skipped
        for name in names:
            try:
                rel = (Path(src_dir) / name).relative_to(lammps_src)
                if str(rel).replace("\\", "/").lower() in blocked_norm:
                    skipped.add(name)
            except ValueError:
                pass
        return skipped

    def _hardlink_or_copy(src: str, dst: str) -> None:
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    shutil.copytree(
        lammps_src, lammps_dest,
        ignore=_ignore,
        copy_function=_hardlink_or_copy,
        symlinks=True,
    )
    logger.info(
        "Filtered LAMMPS copy: %s → %s (blocked %d relpaths)",
        lammps_src, lammps_dest, len(blocked_norm),
    )
    return lammps_dest


def cleanup_filtered_lammps_copy(lammps_copy: Path) -> None:
    """Remove the temp directory created by :func:`create_filtered_lammps_copy`."""
    shutil.rmtree(lammps_copy.parent, ignore_errors=True)

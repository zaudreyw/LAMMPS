#!/usr/bin/env python3
"""Stop-hook self-verification for LAMMPS input authoring tasks.

Fires when the Claude Code agent ends its turn. Checks that
``/workspace/inputs/`` contains at least one ``.in`` file with the minimum
structural requirements for a valid LAMMPS input script. Optionally also
validates by running the LAMMPS binary. If any check fails, emits
``decision: "block"`` on stdout so Claude Code re-enters the agent with the
reason as feedback; otherwise allows the stop.

Validation tiers (in order):
  1. Presence check  — at least one .in file exists.
  2. Structure check — each file contains the required commands: `units`,
                       `atom_style`, and at least one of `run`/`minimize`/
                       `rerun`. Missing any of these almost certainly means
                       the script is incomplete.
  3. LAMMPS check    — if the `lammps` binary is in PATH (and
                       LAMMPS_HOOK_LAMMPS_CHECK=1), run each file with
                       `lammps -in <file> -log none -screen none -nocite`
                       and block on non-zero exit. Note: this will fail if
                       required data files or potential files are absent, so
                       it is disabled by default.

Environment knobs:
  LAMMPS_HOOK_INPUTS_DIR      Override the workspace inputs directory.
                               Defaults to ``$CLAUDE_PROJECT_DIR/inputs``
                               if set, else ``/workspace/inputs``.
  LAMMPS_HOOK_MAX_RETRIES     Max times this hook will block before giving
                               up. Default 2.
  LAMMPS_HOOK_DISABLE         If ``1``/``true``/``yes``, hook no-ops.
  LAMMPS_HOOK_LAMMPS_CHECK    If ``1``/``true``/``yes``, run the lammps
                               binary to validate each script. Off by default
                               because it requires all referenced data files
                               to be present.
  LAMMPS_HOOK_SELF_REFLECT    If ``1``/``true``/``yes``, after checks pass,
                               block once with a self-review prompt.
  LAMMPS_BINARY               Path to the lammps binary. Defaults to
                               ``lammps`` (resolved via PATH).

Input JSON is read from stdin; see Claude Code Stop-hook schema.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MAX_ERRORS_PER_FILE = 8
MAX_FILES_REPORTED = 4

# Commands that are absolutely required in any non-trivial LAMMPS input.
# LAMMPS will error immediately if these are absent.
_REQUIRED_COMMANDS = {"units", "atom_style"}

# At least one of these must be present — otherwise the script defines a
# setup but never does anything.
_RUN_COMMANDS = {"run", "minimize", "rerun", "tad", "neb", "prd", "temper",
                 "temper/npt", "temper/grem", "dynamical_matrix",
                 "dynamical_matrix/vibrational_modes", "server"}


def _envflag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _inputs_dir() -> Path:
    override = os.environ.get("LAMMPS_HOOK_INPUTS_DIR")
    if override:
        return Path(override)
    project = os.environ.get("CLAUDE_PROJECT_DIR")
    if project:
        return Path(project) / "inputs"
    return Path("/workspace/inputs")


def _event_log_path(inputs_dir: Path) -> Path:
    override = os.environ.get("LAMMPS_HOOK_EVENTS_PATH")
    if override:
        return Path(override)
    parent = inputs_dir.parent if inputs_dir.parent.exists() else Path("/tmp")
    return parent / ".verify_hook_events.jsonl"


def _log_event(
    inputs_dir: Path,
    decision: str,
    reason_category: str,
    retries_so_far: int,
    detail: str = "",
) -> None:
    path = _event_log_path(inputs_dir)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "reason_category": reason_category,
        "retries_so_far": retries_so_far,
        "detail": detail,
    }
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def _allow_stop(
    inputs_dir: Path | None = None,
    reason_category: str = "allow",
    retries_so_far: int = 0,
    extra: dict | None = None,
) -> None:
    if inputs_dir is not None:
        _log_event(inputs_dir, "allow", reason_category, retries_so_far)
    payload: dict = {"continue": True, "suppressOutput": True}
    if extra:
        payload.update(extra)
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    sys.exit(0)


def _block(
    reason: str,
    inputs_dir: Path,
    reason_category: str,
    retries_so_far: int,
    detail: str = "",
) -> None:
    _log_event(inputs_dir, "block", reason_category, retries_so_far, detail)
    payload = {"decision": "block", "reason": reason}
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    sys.exit(0)


def _retry_counter(inputs_dir: Path) -> Path:
    parent = inputs_dir.parent if inputs_dir.parent.exists() else Path("/tmp")
    return parent / ".verify_retry_count"


def _bump_counter(counter: Path) -> int:
    try:
        current = int(counter.read_text().strip() or "0")
    except (FileNotFoundError, ValueError):
        current = 0
    current += 1
    try:
        counter.write_text(str(current))
    except OSError:
        pass
    return current


def _list_in_files(inputs_dir: Path) -> list[Path]:
    if not inputs_dir.exists():
        return []
    return sorted(p for p in inputs_dir.rglob("*.in") if p.is_file())


def _extract_commands(path: Path) -> set[str]:
    """Return the set of top-level command names found in a LAMMPS script.

    Strips comments (# to end of line) and blank lines. The command name is
    the first whitespace-delimited token on each non-blank, non-comment line.
    Does not attempt to resolve variables or if/loop constructs.
    """
    commands: set[str] = set()
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return commands
    for line in text.splitlines():
        # Strip inline comments
        line = line.split("#")[0].strip()
        if not line:
            continue
        token = line.split()[0].lower()
        commands.add(token)
    return commands


def _structure_errors(path: Path) -> list[str]:
    """Return a list of structural problems with a LAMMPS input script.

    Returns an empty list if the script looks complete enough to run.
    """
    commands = _extract_commands(path)
    if not commands:
        return ["File is empty or contains only comments."]

    errors: list[str] = []
    for cmd in sorted(_REQUIRED_COMMANDS):
        if cmd not in commands:
            errors.append(
                f"Missing required command `{cmd}`. "
                f"Every LAMMPS script must define `{cmd}` before running."
            )
    if not (_RUN_COMMANDS & commands):
        errors.append(
            "No run or minimize command found. The script will not execute any "
            "simulation. Add `run <N>` (for MD) or `minimize <etol> <ftol> "
            "<maxiter> <maxeval>` (for energy minimization) at the end."
        )
    return errors


def _lammps_validate(
    paths: list[Path],
    inputs_dir: Path,
    binary: str = "lammps",
) -> str | None:
    """Run each .in file through the LAMMPS binary to catch parse errors.

    Returns a formatted error string if any file fails, or None if all pass
    (or if the binary is unavailable). Exits with zero only when LAMMPS
    completes without error.

    Caveat: LAMMPS will fail at runtime if referenced data files or potential
    files are absent. Failures caused by missing external files are reported
    but distinguished from syntax errors so the agent knows what to fix.
    """
    if shutil.which(binary) is None:
        return None

    files_with_errors: list[tuple[Path, list[str]]] = []
    for p in paths:
        try:
            res = subprocess.run(
                [binary, "-in", str(p), "-log", "none", "-screen", "none", "-nocite"],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, OSError):
            continue
        if res.returncode == 0:
            continue

        # Parse LAMMPS stderr/stdout for the first error line.
        # LAMMPS errors look like: "ERROR: <message> (src/...:line)"
        err_lines: list[str] = []
        combined = res.stdout + "\n" + res.stderr
        for line in combined.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith("ERROR") or line.startswith("WARNING"):
                err_lines.append(line)
            if len(err_lines) >= MAX_ERRORS_PER_FILE:
                break
        if not err_lines:
            # Generic failure without parsed errors
            err_lines = [f"lammps exited with code {res.returncode}"]
        files_with_errors.append((p, err_lines))

    if not files_with_errors:
        return None

    parts: list[str] = []
    for p, errs in files_with_errors[:MAX_FILES_REPORTED]:
        try:
            rel = p.relative_to(inputs_dir)
        except (ValueError, AttributeError):
            rel = p
        joined = "\n  ".join(errs)
        parts.append(f"- {rel}:\n  {joined}")
    extra = len(files_with_errors) - MAX_FILES_REPORTED
    summary = "\n".join(parts)
    if extra > 0:
        summary += f"\n- ...plus {extra} more file(s) with errors."
    return summary


def main() -> None:
    inputs_dir = _inputs_dir()

    if _envflag("LAMMPS_HOOK_DISABLE"):
        _allow_stop(inputs_dir, reason_category="disabled")

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        _allow_stop(inputs_dir, reason_category="bad_hook_input")

    counter = _retry_counter(inputs_dir)
    max_retries = int(os.environ.get("LAMMPS_HOOK_MAX_RETRIES", "2") or 2)

    in_files = _list_in_files(inputs_dir)

    # --- Tier 1: presence ---
    if not in_files:
        retries = _bump_counter(counter)
        if retries > max_retries:
            _allow_stop(inputs_dir, reason_category="no_in_max_retries", retries_so_far=retries)
        _block(
            "Stop blocked by verify_outputs hook: no .in files found under "
            f"{inputs_dir}. This is a required output of the task. Write the "
            "LAMMPS input script to "
            f"{inputs_dir}/<name>.in using the Write tool, then end your turn.",
            inputs_dir=inputs_dir,
            reason_category="no_in",
            retries_so_far=retries,
        )

    # --- Tier 2: structure check ---
    all_structure_errors: list[tuple[Path, list[str]]] = []
    for p in in_files:
        errs = _structure_errors(p)
        if errs:
            all_structure_errors.append((p, errs))

    if all_structure_errors:
        retries = _bump_counter(counter)
        if retries > max_retries:
            _allow_stop(
                inputs_dir,
                reason_category="structure_error_max_retries",
                retries_so_far=retries,
            )
        parts: list[str] = []
        for p, errs in all_structure_errors[:MAX_FILES_REPORTED]:
            try:
                rel = p.relative_to(inputs_dir)
            except (ValueError, AttributeError):
                rel = p
            joined = "\n  ".join(errs)
            parts.append(f"- {rel}:\n  {joined}")
        extra = len(all_structure_errors) - MAX_FILES_REPORTED
        summary = "\n".join(parts)
        if extra > 0:
            summary += f"\n- ...plus {extra} more file(s) with structural issues."
        _block(
            "Stop blocked by verify_outputs hook: one or more LAMMPS input "
            f"scripts under {inputs_dir} are structurally incomplete.\n\n"
            f"{summary}\n\n"
            "Fix the issues above and end your turn. Remember that LAMMPS "
            "commands are order-sensitive: `units` and `atom_style` must come "
            "before any geometry or potential commands, and `run`/`minimize` "
            "must come last.",
            inputs_dir=inputs_dir,
            reason_category="structure_error",
            retries_so_far=retries,
            detail=summary[:500],
        )

    # --- Tier 3: optional LAMMPS binary check ---
    if _envflag("LAMMPS_HOOK_LAMMPS_CHECK"):
        binary = os.environ.get("LAMMPS_BINARY", "lammps")
        feedback = _lammps_validate(in_files, inputs_dir, binary=binary)
        if feedback is not None:
            retries = _bump_counter(counter)
            if retries > max_retries:
                _allow_stop(
                    inputs_dir,
                    reason_category="lammps_error_max_retries",
                    retries_so_far=retries,
                )
            _block(
                "Stop blocked by verify_outputs hook: the LAMMPS binary "
                f"reported errors in one or more scripts under {inputs_dir}.\n\n"
                f"{feedback}\n\n"
                "If the error is about a missing data file or potential file, "
                "make sure you have written all referenced files. If it is a "
                "syntax error, correct the command syntax against the LAMMPS "
                "documentation and re-validate with:\n"
                f"  {binary} -in <file>.in -log none -screen none -nocite\n"
                "before ending your turn.",
                inputs_dir=inputs_dir,
                reason_category="lammps_error",
                retries_so_far=retries,
                detail=feedback[:500],
            )

    # --- Optional self-reflection pass ---
    if _envflag("LAMMPS_HOOK_SELF_REFLECT"):
        flag = counter.parent / ".verify_reflected"
        if not flag.exists():
            try:
                flag.write_text("1")
            except OSError:
                pass
            files = ", ".join(
                str(p.relative_to(inputs_dir)) if p.is_relative_to(inputs_dir) else str(p)
                for p in in_files
            )
            _block(
                "Stop blocked by verify_outputs hook (self-reflection pass): "
                f"you produced {files}. Before ending the turn, re-read each "
                "file and verify: (a) the units and atom_style match the "
                "physics described in the task; (b) all pair_coeff lines "
                "reference the correct atom types defined earlier; (c) any "
                "referenced data files or potential files are also written to "
                "/workspace/inputs/; (d) the run length / timestep make sense "
                "for the system. Fix any issues, then end your turn. "
                "This reflection will not repeat.",
                inputs_dir=inputs_dir,
                reason_category="self_reflect",
                retries_so_far=0,
            )

    _allow_stop(inputs_dir, reason_category="in_clean")


if __name__ == "__main__":
    main()

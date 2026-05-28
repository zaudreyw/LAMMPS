# /// script
# dependencies = [
#   "mcp>=1.0.0,<2",
# ]
# ///
"""MCP server exposing a LAMMPS input-script validation tool.

Analogous to xmllint_mcp.py for the GEOS harness: exposes a single tool the
agent can call proactively mid-task to pre-validate a LAMMPS .in script
instead of waiting for the end-of-turn Stop hook.

Two validation tiers run in sequence:

  Tier 1 (always): structural check — verifies that required commands
  (``units``, ``atom_style``) and at least one run/minimize command are
  present in the file.

  Tier 2 (when lammps binary available): invokes
  ``lammps -skiprun -nocite -log none -screen none -in <file>`` to catch
  parse and syntax errors without actually executing the simulation.
  The ``-skiprun`` flag was introduced in LAMMPS 15Dec2020; if the flag is
  not recognised the validator falls back to Tier 1 only.

  Note: LAMMPS will also fail at Tier 2 if referenced data files or
  potential files are absent from /workspace/inputs/.  Those failures are
  reported as "missing file" errors rather than syntax errors so the agent
  can distinguish what to fix.

Tool name (with the ``mcp__lammps-validate__`` prefix Claude Code applies):

    mcp__lammps-validate__validate_lammps_input(in_path: str) -> str

The LAMMPS binary is resolved from the ``LAMMPS_BINARY`` env var or from
PATH (default name ``lammps``). The workspace directory is configurable via
``LAMMPS_VALIDATE_WORKSPACE_DIR`` (defaults to ``/workspace``).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Allow importing lammps_grammar_validate from the same scripts/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lammps_grammar_validate import load_grammar, grammar_errors as _grammar_errors

from mcp.server.fastmcp import FastMCP

_GRAMMAR = load_grammar()

DEFAULT_WORKSPACE = Path(os.environ.get("LAMMPS_VALIDATE_WORKSPACE_DIR", "/workspace"))
INPUTS_DIR = DEFAULT_WORKSPACE / "inputs"

# Commands that are absolutely required in any non-trivial LAMMPS input.
_REQUIRED_COMMANDS = {"units", "atom_style"}
# At least one of these must be present.
_RUN_COMMANDS = {
    "run", "minimize", "rerun", "tad", "neb", "prd", "temper",
    "temper/npt", "temper/grem", "dynamical_matrix", "server",
}

# LAMMPS error/warning output patterns
_MISSING_FILE_RE = re.compile(
    r"(cannot open|no such file|file not found|cannot read)",
    re.IGNORECASE,
)

mcp = FastMCP("lammps-validate")


def _resolve(in_path: str) -> Path:
    p = Path(in_path)
    if p.is_absolute():
        return p
    for candidate in (Path.cwd() / p, DEFAULT_WORKSPACE / p, INPUTS_DIR / p):
        if candidate.exists():
            return candidate
    return p  # let caller emit the not-found error


def _extract_commands(path: Path) -> set[str]:
    commands: set[str] = set()
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return commands
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        token = line.split()[0].lower()
        commands.add(token)
    return commands


def _structural_errors(path: Path) -> list[str]:
    commands = _extract_commands(path)
    if not commands:
        return ["File is empty or contains only comments."]
    errors: list[str] = []
    for cmd in sorted(_REQUIRED_COMMANDS):
        if cmd not in commands:
            errors.append(
                f"Missing required command `{cmd}`. Every LAMMPS input script "
                f"must define `{cmd}` before any geometry or potential commands."
            )
    if not (_RUN_COMMANDS & commands):
        errors.append(
            "No run or minimize command found. Add `run <N>` (MD) or "
            "`minimize <etol> <ftol> <maxiter> <maxeval>` (energy minimization) "
            "at the end of the script."
        )
    return errors


def _binary_validate(target: Path) -> str | None:
    """Run LAMMPS with -skiprun to catch parse errors. Returns error string or None."""
    binary = os.environ.get("LAMMPS_BINARY", "lammps")
    if shutil.which(binary) is None:
        return None

    try:
        res = subprocess.run(
            [binary, "-skiprun", "-nocite", "-log", "none", "-screen", "none", "-in", str(target)],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return f"ERROR: LAMMPS timed out after 60s on {target.name}"
    except OSError as exc:
        return f"ERROR: could not run LAMMPS binary: {exc}"

    if res.returncode == 0:
        return None

    # Check if -skiprun is unsupported (very old LAMMPS)
    combined = res.stdout + "\n" + res.stderr
    if "unknown command" in combined.lower() and "skiprun" in combined.lower():
        return None  # fall back to structural-only; binary check unavailable

    # Classify errors: syntax vs missing-file
    syntax_errs: list[str] = []
    missing_errs: list[str] = []
    for line in combined.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("ERROR") or line.startswith("WARNING"):
            if _MISSING_FILE_RE.search(line):
                missing_errs.append(line)
            else:
                syntax_errs.append(line)
        if len(syntax_errs) + len(missing_errs) >= 12:
            break

    if not syntax_errs and not missing_errs:
        return f"LAMMPS exited with code {res.returncode} (no ERROR lines found)"

    parts: list[str] = []
    if syntax_errs:
        parts.append("Syntax/parse errors (fix the command):")
        parts.extend(f"  {e}" for e in syntax_errs[:8])
    if missing_errs:
        parts.append(
            "Missing-file errors (write the file to /workspace/inputs/ "
            "or fix the path):"
        )
        parts.extend(f"  {e}" for e in missing_errs[:4])
    return "\n".join(parts)


@mcp.tool()
def validate_lammps_input(in_path: str) -> str:
    """Validate a LAMMPS .in script for structural completeness and parse errors.

    Use this BEFORE finishing your turn on every .in file you produced. It
    catches the most common error classes: missing required commands, no run
    command, and LAMMPS parse/syntax errors.

    Tier 1 (always): structural check — verifies ``units``, ``atom_style``,
    and at least one ``run``/``minimize`` command are present.

    Tier 2 (when ``lammps`` binary is in PATH): runs
    ``lammps -skiprun -nocite -log none -screen none -in <file>`` to catch
    syntax errors without executing the simulation. Reports missing-file
    errors separately from syntax errors so you know what to fix.

    Args:
        in_path: Path to the .in file. Absolute (``/workspace/inputs/foo.in``)
            or relative to the workspace (``inputs/foo.in`` / ``foo.in``).

    Returns:
        ``"OK: <file>"`` when both tiers pass, or a multi-line block listing
        each issue found. Fix all reported issues before ending your turn.
    """
    target = _resolve(in_path)
    if not target.exists():
        return f"ERROR: file not found: {target} (resolved from {in_path!r})"

    # Tier 1: structural check
    struct_errs = _structural_errors(target)
    if struct_errs:
        body = "\n".join(f"  - {e}" for e in struct_errs)
        return (
            f"{target}: FAILS structural validation\n"
            f"{body}\n"
            "Fix these issues before running the LAMMPS binary check."
        )

    # Tier 1.5: grammar check — unknown/misspelled command and style names
    grammar_errs = _grammar_errors(target, _GRAMMAR)
    if grammar_errs:
        body = "\n".join(f"  - {e}" for e in grammar_errs)
        return (
            f"{target}: FAILS grammar validation\n"
            f"{body}\n"
            "Fix the unknown commands or style names above before re-validating."
        )

    # Tier 2: binary check
    binary_result = _binary_validate(target)
    if binary_result is None:
        # Either binary not available (structural pass is sufficient) or skiprun succeeded.
        pass
    elif binary_result.startswith("ERROR:"):
        # Infrastructure error; report but don't block
        return (
            f"{target}: structural OK; binary check skipped ({binary_result})\n"
            "Manually verify the script is correct before submitting."
        )
    else:
        return (
            f"{target}: FAILS LAMMPS binary validation\n"
            f"{binary_result}\n"
            "Fix the issues above, then re-validate."
        )

    return f"{target}: OK"


if __name__ == "__main__":
    mcp.run()

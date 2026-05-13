"""Prompt construction helpers for the GEOS eval harness.

Long prompt strings live next to this module in plain ``.txt`` files so the
Python source stays readable. The helper functions here just glue them
together — no semantic changes from the original
``scripts/run_experiment.py`` ``build_*`` / ``*_retry_prompt`` functions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..constants import CONTAINER_GEOS_PRIMER_PATH, RUN_ASSETS_DIR

_PROMPTS_DIR = Path(__file__).parent


def _load(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text()


def _envflag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_RAG_INSTRUCTIONS_PLUGIN = _load("rag_instructions.txt")
_RAG_INSTRUCTIONS_VANILLA = _load("rag_vanilla.txt")
_RAG_INSTRUCTIONS_LAMMPS_PLUGIN = _load("rag_instructions_lammps.txt")
_MEMORY_INSTRUCTIONS = _load("memory_instructions.txt")
_REAL_TOOL_TAIL = _load("real_tool_tail.txt")
_NATIVE_PLUGIN_PREFIX = _load("native_plugin_prefix.txt")
_NATIVE_PLUGIN_PREFIX_LAMMPS = _load("native_plugin_prefix_lammps.txt")
_PSEUDO_TOOL_RETRY = _load("pseudo_tool_retry.txt")
_NO_OUTPUTS_RETRY = _load("no_outputs_retry.txt")
_MISSING_RAG_DISCLAIMER = _load("missing_rag_disclaimer.txt")


def load_agents_md(
    strip_baked_primer: bool = False,
    agents_md_path: Path | None = None,
) -> str:
    """Load an AGENTS.md file.

    ``agents_md_path`` lets the caller point at any AGENTS.md variant
    (e.g. ``run/AGENTS_lammps.md`` for the LAMMPS harness). Defaults to
    ``run/AGENTS.md`` for backwards compatibility.

    ``strip_baked_primer=True`` drops the first section whose heading starts
    with ``# `` followed by ``Primer`` (case-insensitive) and everything
    after it, allowing an external primer file to be injected instead.
    """
    path = agents_md_path if agents_md_path is not None else RUN_ASSETS_DIR / "AGENTS.md"
    if not path.exists():
        raise FileNotFoundError(f"AGENTS.md not found at {path}")
    text = path.read_text()
    if strip_baked_primer:
        # Support both "# GEOS Primer" and "# LAMMPS Primer" headings.
        import re
        m = re.search(r"\n# [^\n]*Primer", text, re.IGNORECASE)
        if m:
            text = text[: m.start()].rstrip() + "\n"
    return text


def load_task_instructions(task_dir: Path) -> str:
    path = task_dir / "instructions.txt"
    if not path.exists():
        raise FileNotFoundError(f"instructions.txt not found in {task_dir}")
    return path.read_text()


def build_prompt(agents_context: str, task_instructions: str) -> str:
    return (
        f"{agents_context}\n\n"
        "--- BEGIN SIMULATION SPECIFICATION ---\n"
        f"{task_instructions.strip()}\n"
        "--- END SIMULATION SPECIFICATION ---"
    )


def build_task_prompt(task_instructions: str) -> str:
    return (
        "--- BEGIN SIMULATION SPECIFICATION ---\n"
        f"{task_instructions.strip()}\n"
        "--- END SIMULATION SPECIFICATION ---"
    )


_SUPERVISOR_INSTRUCTIONS = (
    "\n## Asking the human researcher\n"
    "Some details of the simulation specification you received are "
    "intentionally unspecified. A simulated human researcher is "
    "available via the `mcp__geos-supervisor__consult_supervisor` "
    "tool. Use it when a value or design decision you need is missing "
    "from the brief AND you cannot reasonably infer it from GEOS "
    "conventions, GEOS example simulations, or standard geophysics "
    "practice. Each call costs the researcher's time, so prefer to "
    "infer when you can. Ask short, specific, single-question queries; "
    "batch related parameters into one question when possible. The "
    "researcher will answer concisely using only the original "
    "specification.\n"
)


def build_system_prompt(
    agents_context: str,
    geos_primer_path: Path,
    cheatsheet_path: Path | None = None,
    cheatsheet_in_workspace: bool = False,
    memory_enabled: bool = False,
    memory_prompt_hint: bool = True,
    plugin_enabled: bool = True,
    rag_enabled: bool | None = None,
    supervisor_enabled: bool = False,
    lammps_mode: bool = False,
) -> tuple[str, bool]:
    if rag_enabled is None:
        rag_enabled = plugin_enabled
    primer_text = ""
    primer_inlined = False
    if lammps_mode:
        # LAMMPS: primer is baked into AGENTS_lammps.md; never inject the GEOS primer.
        if "# LAMMPS Primer" in agents_context:
            primer_inlined = True
    elif geos_primer_path.exists() and "# GEOS Primer" not in agents_context:
        primer_text = (
            "\n\n---\n"
            "# GEOS Primer\n\n"
            f"{geos_primer_path.read_text().strip()}\n"
        )
        primer_inlined = True
    elif "# GEOS Primer" in agents_context:
        primer_inlined = True

    cheatsheet_text = ""
    if cheatsheet_path is not None and Path(cheatsheet_path).exists():
        if cheatsheet_in_workspace:
            # Just a pointer; content lives in /workspace/CHEATSHEET.md
            cheatsheet_text = (
                "\n\n---\n"
                "A task-authoring cheatsheet is available at "
                "`/workspace/CHEATSHEET.md` with shortcuts and common pitfalls. "
                "Read it early.\n"
            )
        else:
            body = Path(cheatsheet_path).read_text().strip()
            if body:
                cheatsheet_text = f"\n\n---\n{body}\n"

    if lammps_mode:
        rag_instructions = _RAG_INSTRUCTIONS_LAMMPS_PLUGIN if rag_enabled else ""
    else:
        rag_instructions = _RAG_INSTRUCTIONS_PLUGIN if rag_enabled else _RAG_INSTRUCTIONS_VANILLA

    memory_instructions = (
        _MEMORY_INSTRUCTIONS
        if (plugin_enabled and memory_enabled and memory_prompt_hint) else ""
    )

    supervisor_instructions = (
        _SUPERVISOR_INSTRUCTIONS if supervisor_enabled else ""
    )

    # Optional disclaimer that names the geos-rag MCP tools as unavailable.
    # Mitigates the minimax-m2.7 pseudo-MCP-tool-call failure mode observed
    # on the 2026-05-03 cross-model run when only xmllint MCP is registered
    # (no geos-rag) and minimax hallucinates calls to mcp__geos-rag__*.
    # Off by default to preserve autocamp-experiment-state parity; opt in
    # via env GEOS_PROMPT_DISCLAIM_MISSING_RAG=1 when running cross-model.
    missing_rag_disclaimer = (
        f"\n\n{_MISSING_RAG_DISCLAIMER}\n"
        if (
            _envflag("GEOS_PROMPT_DISCLAIM_MISSING_RAG")
            and not rag_enabled
        )
        else ""
    )

    return (
        f"{agents_context.strip()}{primer_text}{cheatsheet_text}\n\n"
        "---\n"
        + rag_instructions
        + memory_instructions
        + supervisor_instructions
        + _REAL_TOOL_TAIL
        + missing_rag_disclaimer
    ), primer_inlined


def native_plugin_prefix(lammps_mode: bool = False) -> str:
    """Prefix prepended to the prompt when the plugin is enabled (native runner)."""
    return _NATIVE_PLUGIN_PREFIX_LAMMPS if lammps_mode else _NATIVE_PLUGIN_PREFIX


def pseudo_tool_retry_prompt(previous_status: str, counts: dict[str, Any]) -> str:
    pseudo_counts = counts.get("pseudo_tool_counts", {})
    pseudo_summary = ", ".join(
        f"{name} x{count}" for name, count in sorted(pseudo_counts.items())
    ) or "unknown pseudo tool"
    return _PSEUDO_TOOL_RETRY.format(
        previous_status=previous_status,
        pseudo_summary=pseudo_summary,
    )


def no_outputs_retry_prompt(previous_status: str) -> str:
    return _NO_OUTPUTS_RETRY.format(previous_status=previous_status)


def redact_command_for_display(cmd: list[str]) -> str:
    redacted: list[str] = []
    secret_markers = ("KEY=", "TOKEN=", "SECRET=", "PASSWORD=")
    prompt_flags = {"--append-system-prompt", "--system-prompt"}
    previous = ""
    for token in cmd:
        if previous in prompt_flags:
            redacted.append("<system_prompt>")
            previous = token
            continue
        if any(marker in token for marker in secret_markers):
            key = token.split("=", 1)[0]
            redacted.append(f"{key}=<redacted>")
        else:
            redacted.append(token)
        previous = token
    return " ".join(redacted)


__all__ = [
    "build_prompt",
    "build_task_prompt",
    "build_system_prompt",
    "load_agents_md",
    "load_task_instructions",
    "native_plugin_prefix",
    "pseudo_tool_retry_prompt",
    "no_outputs_retry_prompt",
    "redact_command_for_display",
]

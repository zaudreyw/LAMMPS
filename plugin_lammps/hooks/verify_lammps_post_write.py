#!/usr/bin/env python3
"""PostToolUse hook: check LAMMPS .in files right after Write|Edit|MultiEdit.

Catches structural problems (missing required commands, no run command) within
seconds of a bad write instead of waiting until end-of-turn for the Stop hook.

Behavior:
  - Only fires for Write/Edit/MultiEdit on .in files inside
    $LAMMPS_HOOK_INPUTS_DIR (default $CLAUDE_PROJECT_DIR/inputs or
    /workspace/inputs).
  - Allows (no output) on success.
  - Returns ``decision: "block"`` with a concise fix hint on failure.
  - No retry budget — every problematic write fires until the file is valid.
  - Honors LAMMPS_HOOK_DISABLE.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import importlib.util
import sys as _sys

# Load verify_outputs from the same hooks/ directory without requiring it
# to be on sys.path or installed as a package.
_hooks_dir = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "verify_outputs", _hooks_dir / "verify_outputs.py"
)
_vo = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_vo)  # type: ignore[union-attr]

_envflag = _vo._envflag
_extract_commands = _vo._extract_commands
_REQUIRED_COMMANDS = _vo._REQUIRED_COMMANDS
_RUN_COMMANDS = _vo._RUN_COMMANDS


def _inputs_dir() -> Path:
    override = os.environ.get("LAMMPS_HOOK_INPUTS_DIR")
    if override:
        return Path(override)
    project = os.environ.get("CLAUDE_PROJECT_DIR")
    if project:
        return Path(project) / "inputs"
    return Path("/workspace/inputs")


def _event_log_path(inputs_dir: Path) -> Path:
    override = os.environ.get("LAMMPS_POST_HOOK_EVENTS_PATH")
    if override:
        return Path(override)
    parent = inputs_dir.parent if inputs_dir.parent.exists() else Path("/tmp")
    return parent / ".verify_post_hook_events.jsonl"


def _log_event(inputs_dir: Path, decision: str, detail: str = "") -> None:
    path = _event_log_path(inputs_dir)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "detail": detail,
    }
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def main() -> None:
    inputs_dir = _inputs_dir()

    if _envflag("LAMMPS_HOOK_DISABLE"):
        json.dump({"continue": True, "suppressOutput": True}, sys.stdout)
        sys.stdout.write("\n")
        return

    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError:
        json.dump({"continue": True, "suppressOutput": True}, sys.stdout)
        sys.stdout.write("\n")
        return

    # Extract the file path from the tool result
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input", {})

    file_path_str: str | None = None
    if tool_name == "Write":
        file_path_str = tool_input.get("file_path")
    elif tool_name in ("Edit", "MultiEdit"):
        file_path_str = tool_input.get("file_path")

    if not file_path_str:
        json.dump({"continue": True, "suppressOutput": True}, sys.stdout)
        sys.stdout.write("\n")
        return

    file_path = Path(file_path_str)

    # Only check .in files inside the inputs directory
    if file_path.suffix.lower() != ".in":
        json.dump({"continue": True, "suppressOutput": True}, sys.stdout)
        sys.stdout.write("\n")
        return

    try:
        if not file_path.is_relative_to(inputs_dir):
            json.dump({"continue": True, "suppressOutput": True}, sys.stdout)
            sys.stdout.write("\n")
            return
    except (ValueError, AttributeError):
        pass

    if not file_path.exists():
        json.dump({"continue": True, "suppressOutput": True}, sys.stdout)
        sys.stdout.write("\n")
        return

    commands = _extract_commands(file_path)
    if not commands:
        _log_event(inputs_dir, "block", "empty file")
        json.dump(
            {
                "decision": "block",
                "reason": (
                    f"PostToolUse hook: {file_path.name} appears to be empty or contains "
                    "only comments. Write the LAMMPS input script content."
                ),
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
        return

    missing_required = sorted(_REQUIRED_COMMANDS - commands)
    has_run = bool(_RUN_COMMANDS & commands)

    issues: list[str] = []
    if missing_required:
        issues.append(f"missing required command(s): {', '.join(missing_required)}")
    if not has_run:
        issues.append("no run/minimize command found — script will not execute")

    if issues:
        detail = "; ".join(issues)
        _log_event(inputs_dir, "block", detail)
        json.dump(
            {
                "decision": "block",
                "reason": (
                    f"PostToolUse hook: {file_path.name} is structurally incomplete — "
                    f"{detail}. Fix before ending your turn."
                ),
            },
            sys.stdout,
        )
        sys.stdout.write("\n")
        return

    _log_event(inputs_dir, "allow")
    json.dump({"continue": True, "suppressOutput": True}, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

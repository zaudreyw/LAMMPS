"""Path / regex / flag constants used across :mod:`runner`.

Mirrors lines 91-135 of the original ``scripts/run_experiment.py``.
"""

from __future__ import annotations

import re
import subprocess
import threading
from pathlib import Path

# This module lives at src/runner/constants.py, so REPO_ROOT is two
# parents up (repo3/src/runner/ -> repo3/src/ -> repo3/).
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts"  # legacy alias; preserved for parity
RUN_ASSETS_DIR = REPO_ROOT / "run"  # AGENTS.md + Dockerfile live here
DATA_DIR = REPO_ROOT / "data"
EXPERIMENTS_DIR = DATA_DIR / "eval" / "experiments"
GROUND_TRUTH_DIR = DATA_DIR / "eval" / "experiments_gt"
DEFAULT_GEOS_LIB_DIR = Path("/data/shared/geophysics_agent_data/data/GEOS")
GEOS_LIB_DIR = DEFAULT_GEOS_LIB_DIR
# Filtered GEOS trees (hardlink farms) are created here. Must be writable and on
# the same filesystem as --geos-lib-dir for efficient hardlinks (see contamination.py).
TEMP_GEOS_PARENT = Path("/data/shared/geophysics_agent_data/data/eval/tmp_geos")
DOCKER_IMAGE = "geos-eval"
DEFAULT_PLUGIN_DIR = REPO_ROOT / "plugin"  # .claude-plugin/plugin.json lives under plugin/

DEFAULT_VECTOR_DB_DIR = Path("/data/shared/geophysics_agent_data/data/vector_db")
DEFAULT_GEOS_PRIMER_PATH = Path(
    "/home/brianliu/geophys-embodied-agent-framework/modules/profile/GEOS_PRIMER.md"
)
DEFAULT_CLAUDE_MODEL = "minimax/minimax-m2.7"
CONTAINER_PLUGIN_DIR = Path("/plugins/repo3")
CONTAINER_SETTINGS_PATH = Path("/workspace/claude_settings.json")
CONTAINER_VECTOR_DB_DIR = Path("/data/shared/geophysics_agent_data/data/vector_db")
CONTAINER_MCP_CONFIG_PATH = Path("/workspace/claude_mcp_config.json")
CONTAINER_GEOS_PRIMER_PATH = Path("/workspace/GEOS_PRIMER.md")
RAG_TOOL_NAMES = {"search_navigator", "search_schema", "search_technical"}
PSEUDO_TOOL_RE = re.compile(r"invoke\s+name=[\"']([^\"']+)[\"']", re.IGNORECASE)
NATIVE_CLAUDE_TOOLS = "default"
# Each entry is passed as its own --disallowedTools argument. Skill is blocked
# because the repo3-plugin:geos-rag skill wrapper breaks non-Anthropic providers
# (the RAG instructions are injected directly into the system prompt instead).
# AskUserQuestion is blocked because this harness runs Claude non-interactively
# via `claude -p`; any AskUserQuestion call stalls the turn and is a known
# cause of the premature-end_turn failure mode (see docs/XN-010).
NATIVE_CLAUDE_DISALLOWED_TOOLS = ("Skill", "AskUserQuestion")

DEFAULT_TIMEOUT = 1200  # seconds per task (20 minutes)

# ---------------------------------------------------------------------------
# LAMMPS-specific constants (additive — GEOS paths above are unchanged)
# ---------------------------------------------------------------------------
DEFAULT_LAMMPS_LIB_DIR = Path("/data/shared/lammps_agent_data/data/lammps")
CONTAINER_LAMMPS_LIB_DIR = Path("/lammps_lib")
DEFAULT_LAMMPS_VECTOR_DB_DIR = Path("/data/shared/lammps_agent_data/data/vector_db")
CONTAINER_LAMMPS_VECTOR_DB_DIR = Path("/data/shared/lammps_agent_data/data/vector_db")
LAMMPS_EXPERIMENTS_DIR = DATA_DIR / "eval" / "experiments_lammps"
LAMMPS_RAG_TOOL_NAMES = {"search_navigator", "search_technical", "search_commands"}
# Filtered LAMMPS trees (hardlink farms) are created here.
# Must be on the same filesystem as --lammps-lib-dir for efficient hardlinks.
TEMP_LAMMPS_PARENT = Path("/data/shared/lammps_agent_data/data/eval/tmp_lammps")
DEFAULT_LAMMPS_EXAMPLE_PAIRS = DATA_DIR / "eval" / "lammps_example_pairs.jsonl"

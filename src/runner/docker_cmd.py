"""Docker command building, MCP preflight, vector-DB copy, and primer-file management."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .constants import (
    CONTAINER_GEOS_PRIMER_PATH,
    CONTAINER_LAMMPS_LIB_DIR,
    CONTAINER_LAMMPS_VECTOR_DB_DIR,
    CONTAINER_MCP_CONFIG_PATH,
    CONTAINER_PLUGIN_DIR,
    CONTAINER_SETTINGS_PATH,
    CONTAINER_VECTOR_DB_DIR,
    DOCKER_IMAGE,
    NATIVE_CLAUDE_DISALLOWED_TOOLS,
    NATIVE_CLAUDE_TOOLS,
)


def create_runtime_vector_db_copy(vector_db_src: Path, result_dir: Path) -> Path:
    """Create a writable per-task ChromaDB copy.

    ChromaDB opens its sqlite backing files in a way that can write lock files
    even for read-oriented queries, so the container cannot mount the shared DB
    read-only.  Copying keeps the shared source untouched while allowing each
    parallel task to run independently.
    """
    vector_db_dest = result_dir / ".vector_db_runtime"
    if vector_db_dest.exists():
        shutil.rmtree(vector_db_dest)
    shutil.copytree(vector_db_src, vector_db_dest, symlinks=True)
    return vector_db_dest


def remove_workspace_geos_primer(result_dir: Path) -> None:
    primer_dest = result_dir / CONTAINER_GEOS_PRIMER_PATH.name
    if primer_dest.is_dir():
        shutil.rmtree(primer_dest)
    elif primer_dest.exists() or primer_dest.is_symlink():
        primer_dest.unlink()


def build_claude_native_mcp_smoke_command(
    *,
    result_dir: Path,
    plugin_dir: Path,
    vector_db_dir: Path,
) -> list[str]:
    return [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{result_dir}:/workspace:rw",
        "-v", f"{plugin_dir}:/plugins/repo3:ro",
        "-v", f"{vector_db_dir}:{CONTAINER_VECTOR_DB_DIR}:rw",
        "-e", "HOME=/workspace/.claude_home",
        "-e", "UV_CACHE_DIR=/workspace/.uv_cache",
        "-e", "CLAUDE_PLUGIN_ROOT=/plugins/repo3",
        "-e", f"GEOS_VECTOR_DB_DIR={CONTAINER_VECTOR_DB_DIR}",
        DOCKER_IMAGE,
        "uv",
        "run",
        "--script",
        str(CONTAINER_PLUGIN_DIR / "scripts" / "geos_rag_mcp.py"),
        "--smoke",
    ]


def preflight_claude_native_mcp(
    *,
    result_dir: Path,
    plugin_dir: Path,
    vector_db_dir: Path,
    timeout: int = 180,
) -> dict[str, Any]:
    """Warm the uv script env and prove the repo3 MCP server can open its DB."""
    cmd = build_claude_native_mcp_smoke_command(
        result_dir=result_dir,
        plugin_dir=plugin_dir,
        vector_db_dir=vector_db_dir,
    )
    started = time.time()
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    result = {
        "command": cmd,
        "exit_code": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 1),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "updated": datetime.now().isoformat(),
    }
    (result_dir / "mcp_preflight.json").write_text(json.dumps(result, indent=2))
    if completed.returncode != 0:
        detail = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part.strip()
        )
        raise RuntimeError(
            "repo3 GEOS RAG MCP preflight failed before launching Claude. "
            "Rebuild the geos-eval image if uv is missing, then rerun. "
            f"Details: {detail or 'no output'}"
        )
    return result


def build_claude_native_command(
    *,
    filtered_geos: Path,
    result_dir: Path,
    plugin_dir: Path | None,
    vector_db_dir: Path | None,
    model: str,
    system_prompt: str,
    prompt: str,
    enable_plugin: bool = True,
    supervisor_spec_host_path: Path | None = None,
) -> list[str]:
    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{filtered_geos}:/geos_lib:ro",
        "-v", f"{result_dir}:/workspace:rw",
    ]
    if enable_plugin:
        if plugin_dir is None or vector_db_dir is None:
            raise ValueError("plugin_dir and vector_db_dir required when enable_plugin=True")
        cmd += [
            "-v", f"{plugin_dir}:/plugins/repo3:ro",
            "-v", f"{vector_db_dir}:{CONTAINER_VECTOR_DB_DIR}:rw",
        ]
    if supervisor_spec_host_path is not None:
        # Mounted at a fixed container path consumed by supervisor_mcp.py.
        cmd += [
            "-v",
            f"{supervisor_spec_host_path}:/supervisor/spec.md:ro",
        ]
    cmd += [
        "-e", "HOME=/workspace/.claude_home",
        "-e", "XDG_CONFIG_HOME=/workspace/.claude_home/.config",
        "-e", "UV_CACHE_DIR=/workspace/.uv_cache",
        "-e", "ANTHROPIC_BASE_URL",
        "-e", "ANTHROPIC_AUTH_TOKEN",
        "-e", "ANTHROPIC_API_KEY",
        "-e", "OPENROUTER_API_KEY",
        "-e", "OPENAI_API_KEY",
        "-e", "DEEPSEEK_API_KEY",
        # Forwards for plugin/hooks/verify_outputs.py knobs. Absent vars
        # are fine — hook has sane defaults.
        "-e", "GEOS_HOOK_DISABLE",
        "-e", "GEOS_HOOK_MAX_RETRIES",
        "-e", "GEOS_HOOK_SELF_REFLECT",
        "-e", "GEOS_HOOK_XMLLINT",
        "-e", "GEOS_HOOK_SCHEMA_PATH",
    ]
    if enable_plugin:
        cmd += [
            "-e", "GEOS_VECTOR_DB_DIR",
            "-e", "EXCLUDED_GT_XML_FILENAMES",
            "-e", "EXCLUDED_RST_PATHS",
            # CLAUDE_PLUGIN_ROOT is used by the plugin's hooks.json to locate
            # the hook script (python3 ${CLAUDE_PLUGIN_ROOT}/hooks/verify_outputs.py).
            "-e", f"CLAUDE_PLUGIN_ROOT={CONTAINER_PLUGIN_DIR}",
        ]
    cmd += [
        DOCKER_IMAGE,
        "claude",
        "-p",
        "--verbose",
        "--model", model,
        "--append-system-prompt", system_prompt,
        "--tools", NATIVE_CLAUDE_TOOLS,
    ]
    for disallowed in NATIVE_CLAUDE_DISALLOWED_TOOLS:
        cmd += ["--disallowedTools", disallowed]
    if enable_plugin:
        cmd += [
            f"--mcp-config={CONTAINER_MCP_CONFIG_PATH}",
            "--strict-mcp-config",
            # The Stop hook (verify_outputs.py) is registered via --settings
            # rather than --plugin-dir so the tool list matches pre-hook runs
            # (E17/E18) exactly. Loading the plugin as a plugin would surface
            # its skill in the tool list and confound hook-effect experiments
            # with tool-list-shape effects. See RN-002 / XN-010.
            "--settings", str(CONTAINER_SETTINGS_PATH),
        ]
    cmd += [
        "--output-format", "stream-json",
        "--permission-mode", "bypassPermissions",
        # Separator so a prompt starting with `--` (e.g. the task spec opens
        # with `--- BEGIN SIMULATION SPECIFICATION ---`) isn't parsed as a flag.
        "--",
        prompt,
    ]
    return cmd


def build_claude_native_env(
    *,
    blocked_xml_filenames: list[str],
    blocked_rst_relpaths: list[str],
    vector_db_dir: Path | None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = os.environ.get(
        "ANTHROPIC_BASE_URL",
        "https://openrouter.ai/api",
    )
    if vector_db_dir is not None:
        env["GEOS_VECTOR_DB_DIR"] = str(CONTAINER_VECTOR_DB_DIR)
        env["EXCLUDED_GT_XML_FILENAMES"] = json.dumps(blocked_xml_filenames)
        env["EXCLUDED_RST_PATHS"] = json.dumps(blocked_rst_relpaths)

        # The repo3 MCP server uses OPENROUTER_API_KEY for embeddings.  For this
        # eval path, the OpenRouter Claude auth token is a suitable fallback without
        # putting a secret in the docker command line.
        if not env.get("OPENROUTER_API_KEY") and env.get("ANTHROPIC_AUTH_TOKEN"):
            env["OPENROUTER_API_KEY"] = env["ANTHROPIC_AUTH_TOKEN"]

        # Keep the host path available in metadata/debug logs without overriding the
        # container-visible GEOS_VECTOR_DB_DIR used by the MCP server.
        env["HOST_GEOS_VECTOR_DB_DIR"] = str(vector_db_dir)
    return env


# ---------------------------------------------------------------------------
# LAMMPS-specific docker command builders
# (additive — the GEOS functions above are unchanged)
# ---------------------------------------------------------------------------

def build_lammps_native_command(
    *,
    lammps_lib_dir: Path,
    result_dir: Path,
    plugin_dir: Path | None,
    vector_db_dir: Path | None,
    model: str,
    system_prompt: str,
    prompt: str,
    enable_plugin: bool = True,
) -> list[str]:
    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{lammps_lib_dir}:{CONTAINER_LAMMPS_LIB_DIR}:ro",
        "-v", f"{result_dir}:/workspace:rw",
    ]
    if enable_plugin:
        if plugin_dir is None or vector_db_dir is None:
            raise ValueError("plugin_dir and vector_db_dir required when enable_plugin=True")
        cmd += [
            "-v", f"{plugin_dir}:/plugins/repo3:ro",
            "-v", f"{vector_db_dir}:{CONTAINER_LAMMPS_VECTOR_DB_DIR}:rw",
        ]
    cmd += [
        "-e", "HOME=/workspace/.claude_home",
        "-e", "XDG_CONFIG_HOME=/workspace/.claude_home/.config",
        "-e", "UV_CACHE_DIR=/workspace/.uv_cache",
        "-e", "ANTHROPIC_BASE_URL",
        "-e", "ANTHROPIC_AUTH_TOKEN",
        "-e", "ANTHROPIC_API_KEY",
        "-e", "OPENROUTER_API_KEY",
        "-e", "OPENAI_API_KEY",
        # Forwards for LAMMPS hook knobs
        "-e", "LAMMPS_HOOK_DISABLE",
        "-e", "LAMMPS_HOOK_MAX_RETRIES",
        "-e", "LAMMPS_HOOK_SELF_REFLECT",
        "-e", "LAMMPS_HOOK_LAMMPS_CHECK",
        "-e", "LAMMPS_BINARY",
    ]
    if enable_plugin:
        cmd += [
            "-e", "LAMMPS_VECTOR_DB_DIR",
            "-e", "EXCLUDED_GT_IN_FILENAMES",
            "-e", "EXCLUDED_RST_PATHS",
            "-e", f"CLAUDE_PLUGIN_ROOT={CONTAINER_PLUGIN_DIR}",
        ]
    cmd += [
        DOCKER_IMAGE,
        "claude",
        "-p",
        "--verbose",
        "--model", model,
        "--append-system-prompt", system_prompt,
        "--tools", NATIVE_CLAUDE_TOOLS,
    ]
    for disallowed in NATIVE_CLAUDE_DISALLOWED_TOOLS:
        cmd += ["--disallowedTools", disallowed]
    if enable_plugin:
        cmd += [
            f"--mcp-config={CONTAINER_MCP_CONFIG_PATH}",
            "--strict-mcp-config",
            "--settings", str(CONTAINER_SETTINGS_PATH),
        ]
    cmd += [
        "--output-format", "stream-json",
        "--permission-mode", "bypassPermissions",
        "--",
        prompt,
    ]
    return cmd


def build_lammps_native_env(
    *,
    blocked_in_filenames: list[str],
    blocked_rst_relpaths: list[str],
    vector_db_dir: Path | None,
) -> dict[str, str]:
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = os.environ.get(
        "ANTHROPIC_BASE_URL",
        "https://openrouter.ai/api",
    )
    if vector_db_dir is not None:
        env["LAMMPS_VECTOR_DB_DIR"] = str(CONTAINER_LAMMPS_VECTOR_DB_DIR)
        env["EXCLUDED_GT_IN_FILENAMES"] = json.dumps(blocked_in_filenames)
        env["EXCLUDED_RST_PATHS"] = json.dumps(blocked_rst_relpaths)
        if not env.get("OPENROUTER_API_KEY") and env.get("ANTHROPIC_AUTH_TOKEN"):
            env["OPENROUTER_API_KEY"] = env["ANTHROPIC_AUTH_TOKEN"]
        env["HOST_LAMMPS_VECTOR_DB_DIR"] = str(vector_db_dir)
    return env


def preflight_lammps_mcp(
    *,
    result_dir: Path,
    plugin_dir: Path,
    vector_db_dir: Path,
    timeout: int = 180,
) -> dict[str, Any]:
    """Warm the uv script env and confirm the LAMMPS RAG MCP server can open its DB."""
    cmd = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{result_dir}:/workspace:rw",
        "-v", f"{plugin_dir}:/plugins/repo3:ro",
        "-v", f"{vector_db_dir}:{CONTAINER_LAMMPS_VECTOR_DB_DIR}:rw",
        "-e", "HOME=/workspace/.claude_home",
        "-e", "UV_CACHE_DIR=/workspace/.uv_cache",
        "-e", "CLAUDE_PLUGIN_ROOT=/plugins/repo3",
        "-e", f"LAMMPS_VECTOR_DB_DIR={CONTAINER_LAMMPS_VECTOR_DB_DIR}",
        DOCKER_IMAGE,
        "uv",
        "run",
        "--script",
        str(CONTAINER_PLUGIN_DIR / "scripts" / "lammps_rag_mcp.py"),
        "--smoke",
    ]
    started = time.time()
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )
    result = {
        "command": cmd,
        "exit_code": completed.returncode,
        "elapsed_seconds": round(time.time() - started, 1),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "updated": datetime.now().isoformat(),
    }
    (result_dir / "mcp_preflight.json").write_text(json.dumps(result, indent=2))
    if completed.returncode != 0:
        detail = "\n".join(
            part.strip()
            for part in (completed.stdout, completed.stderr)
            if part.strip()
        )
        raise RuntimeError(
            "LAMMPS RAG MCP preflight failed. "
            "Rebuild the Docker image or check the vector DB path. "
            f"Details: {detail or 'no output'}"
        )
    return result

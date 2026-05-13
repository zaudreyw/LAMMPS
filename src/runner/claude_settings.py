"""claude_settings.json + claude_mcp_config.json writers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .constants import (
    CONTAINER_LAMMPS_VECTOR_DB_DIR,
    CONTAINER_MCP_CONFIG_PATH,
    CONTAINER_PLUGIN_DIR,
    CONTAINER_SETTINGS_PATH,
    CONTAINER_VECTOR_DB_DIR,
    REPO_ROOT,
)


def _envflag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp_path.replace(path)


def write_claude_settings(*, result_dir: Path, hook_enabled: bool) -> Path:
    """Write claude_settings.json with the Stop hook (verify_outputs.py).

    Registering the hook via --settings instead of --plugin-dir keeps the
    tool list identical to pre-hook runs (plugin skill does not surface).
    The GEOS_HOOK_DISABLE env var is the runtime kill switch; passing
    hook_enabled=False here omits the hook from settings entirely for
    maximal baseline parity with E17.

    The optional PostToolUse hook (verify_xml_post_write.py) is registered
    only when the env var ``GEOS_HOOK_POSTTOOLUSE`` is set to a truthy
    value (1/true/yes/on). Default behaviour reproduces the harness state
    at commit `autocamp-experiment-state` (no PostToolUse hook); the new
    behaviour, useful for cross-model runs against `<<TagTag>`-prone
    backbones, is opt-in via the env var. See
    docs/2026-05-03_harness-sequencing.md.
    """
    container_stop_hook = CONTAINER_PLUGIN_DIR / "hooks" / "verify_outputs.py"
    container_post_hook = CONTAINER_PLUGIN_DIR / "hooks" / "verify_xml_post_write.py"
    settings: dict[str, Any] = {}
    if hook_enabled:
        hooks_cfg: dict[str, Any] = {
            "Stop": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {container_stop_hook}",
                            "timeout": 30,
                        }
                    ],
                }
            ],
        }
        if _envflag("GEOS_HOOK_POSTTOOLUSE"):
            # Per-write XML parse check. Catches the `<<TagTag>` failure
            # mode within seconds so the agent can fix via Edit instead of
            # discovering it at end_turn and rewriting whole files under
            # the 40-min budget. Cheap (~50ms per check); 15s timeout is
            # paranoia. Off by default to preserve parity with the
            # autocamp-experiment-state harness (see harness-sequencing doc).
            hooks_cfg["PostToolUse"] = [
                {
                    "matcher": "Write|Edit|MultiEdit",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {container_post_hook}",
                            "timeout": 15,
                        }
                    ],
                }
            ]
        settings["hooks"] = hooks_cfg
    path = result_dir / CONTAINER_SETTINGS_PATH.name
    _safe_write_json(path, settings)
    return path


def write_claude_mcp_config(
    *,
    result_dir: Path,
    blocked_xml_filenames: list[str],
    blocked_rst_relpaths: list[str],
    enable_memory: bool = False,
    enable_noop: bool = False,
    enable_xmllint: bool = False,
    enable_rag: bool = True,
    enable_supervisor: bool = False,
    supervisor_spec_container_path: str = "/supervisor/spec.md",
    supervisor_task_name: str = "",
    supervisor_model: str = "deepseek-v4-flash",
    supervisor_base_url: str = "https://api.deepseek.com/v1",
    memory_variant: str = "lexical",  # "lexical" (memory_mcp.py) or "embed" (memory_mcp_embed.py)
    memory_items_host_path: Path | None = None,
    memory_embed_index_host_path: Path | None = None,
) -> Path:
    """Write the explicit MCP config Claude Code uses inside the container.

    Claude Code loads repo3 plugin skills via --plugin-dir in this eval path, but
    the plugin manifest's mcpServers block is not reliably activated in --bare
    mode.  Passing an explicit --mcp-config keeps the RAG tools available without
    storing API secrets in the workspace file; secrets are supplied through the
    docker process environment.

    If enable_memory=True, additionally register the memory_mcp server that
    serves a frozen G-Memory-lite index from /plugins/repo3/memory_index.json.
    """
    servers: dict[str, Any] = {}
    if enable_rag:
        servers["geos-rag"] = {
            "type": "stdio",
            "command": "uv",
            "args": [
                "run",
                "--script",
                str(CONTAINER_PLUGIN_DIR / "scripts" / "geos_rag_mcp.py"),
            ],
            "env": {
                "CLAUDE_PLUGIN_ROOT": str(CONTAINER_PLUGIN_DIR),
                "GEOS_VECTOR_DB_DIR": str(CONTAINER_VECTOR_DB_DIR),
                "EXCLUDED_GT_XML_FILENAMES": json.dumps(blocked_xml_filenames),
                "EXCLUDED_RST_PATHS": json.dumps(blocked_rst_relpaths),
            },
        }
    if enable_memory:
        if memory_variant == "embed":
            # M3 — embedding MCP with hard-error on missing key + preflight (RN-003 P2 #8).
            # Host paths under plugin/ mount at /plugins/repo3/ in container; translate.
            def _host_plugin_to_container(host_path: Path | None, default_name: str) -> str:
                if host_path is None:
                    return str(CONTAINER_PLUGIN_DIR / default_name)
                hp = Path(host_path).resolve()
                try:
                    rel = hp.relative_to(REPO_ROOT / "plugin")
                    return str(CONTAINER_PLUGIN_DIR / rel)
                except ValueError:
                    # Not under plugin/; leave as-is (may not mount)
                    return str(hp)

            mem_env = {
                "CLAUDE_PLUGIN_ROOT": str(CONTAINER_PLUGIN_DIR),
                "MEMORY_ITEMS_PATH": _host_plugin_to_container(
                    memory_items_host_path, "memory_items.json"),
                "MEMORY_EMBED_INDEX_PATH": _host_plugin_to_container(
                    memory_embed_index_host_path, "memory_items_embeddings.json"),
            }
            servers["memory"] = {
                "type": "stdio",
                "command": "uv",
                "args": [
                    "run",
                    "--script",
                    "--with", "numpy>=1.26",
                    "--with", "requests>=2.31",
                    str(CONTAINER_PLUGIN_DIR / "scripts" / "memory_mcp_embed.py"),
                ],
                "env": mem_env,
            }
        else:
            servers["memory"] = {
                "type": "stdio",
                "command": "uv",
                "args": [
                    "run",
                    "--script",
                    str(CONTAINER_PLUGIN_DIR / "scripts" / "memory_mcp.py"),
                ],
                "env": {
                    "CLAUDE_PLUGIN_ROOT": str(CONTAINER_PLUGIN_DIR),
                    "MEMORY_INDEX_PATH": str(CONTAINER_PLUGIN_DIR / "memory_index.json"),
                },
            }
    if enable_noop:
        servers["noop"] = {
            "type": "stdio",
            "command": "uv",
            "args": [
                "run",
                "--script",
                str(CONTAINER_PLUGIN_DIR / "scripts" / "noop_mcp.py"),
            ],
            "env": {
                "CLAUDE_PLUGIN_ROOT": str(CONTAINER_PLUGIN_DIR),
            },
        }
    if enable_supervisor:
        # Simulated-human supervisor for the interactive-autonomy study.
        # The MCP reads the task's full original specification from a path
        # that lives OUTSIDE /workspace; the path is passed via the per-
        # server env block here, not via the docker `-e` list, so it does
        # not appear in the agent-visible environment. The agent's tools
        # can still in principle Read this path; we rely on the agent
        # not having any reason to look. Telemetry of supervisor calls is
        # written to /workspace/supervisor_calls.jsonl for audit.
        servers["geos-supervisor"] = {
            "type": "stdio",
            "command": "uv",
            "args": [
                "run",
                "--script",
                str(CONTAINER_PLUGIN_DIR / "scripts" / "supervisor_mcp.py"),
            ],
            "env": {
                "CLAUDE_PLUGIN_ROOT": str(CONTAINER_PLUGIN_DIR),
                "SUPERVISOR_SPEC_PATH": supervisor_spec_container_path,
                "SUPERVISOR_TASK_NAME": supervisor_task_name,
                "SUPERVISOR_LLM_MODEL": supervisor_model,
                "SUPERVISOR_LLM_BASE_URL": supervisor_base_url,
                # API key forwarded from the docker process env (see
                # docker_cmd.py — DEEPSEEK_API_KEY is added to -e list when
                # any agent has supervisor_enabled).
                "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", ""),
            },
        }
    if enable_xmllint:
        servers["xmllint"] = {
            "type": "stdio",
            "command": "uv",
            "args": [
                "run",
                "--script",
                str(CONTAINER_PLUGIN_DIR / "scripts" / "xmllint_mcp.py"),
            ],
            "env": {
                "CLAUDE_PLUGIN_ROOT": str(CONTAINER_PLUGIN_DIR),
                # Fixed inside the container — the schema lives next to the
                # filtered GEOS source mount.
                "XMLLINT_SCHEMA_PATH": "/geos_lib/src/coreComponents/schema/schema.xsd",
            },
        }
    mcp_config_path = result_dir / CONTAINER_MCP_CONFIG_PATH.name
    _safe_write_json(mcp_config_path, {"mcpServers": servers})
    return mcp_config_path


# ---------------------------------------------------------------------------
# LAMMPS variants
# (additive — GEOS functions above are unchanged)
# ---------------------------------------------------------------------------

def write_lammps_claude_settings(*, result_dir: Path, hook_enabled: bool) -> Path:
    """Write claude_settings.json wiring the LAMMPS Stop + PostToolUse hooks.

    plugin_lammps/ is mounted at /plugins/repo3 (same container path as the GEOS
    plugin), so CONTAINER_PLUGIN_DIR already resolves correctly.
    """
    container_stop_hook = CONTAINER_PLUGIN_DIR / "hooks" / "verify_outputs.py"
    container_post_hook = CONTAINER_PLUGIN_DIR / "hooks" / "verify_lammps_post_write.py"
    settings: dict[str, Any] = {}
    if hook_enabled:
        hooks_cfg: dict[str, Any] = {
            "Stop": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {container_stop_hook}",
                            "timeout": 30,
                        }
                    ],
                }
            ],
        }
        if _envflag("LAMMPS_HOOK_POSTTOOLUSE"):
            hooks_cfg["PostToolUse"] = [
                {
                    "matcher": "Write|Edit|MultiEdit",
                    "hooks": [
                        {
                            "type": "command",
                            "command": f"python3 {container_post_hook}",
                            "timeout": 15,
                        }
                    ],
                }
            ]
        settings["hooks"] = hooks_cfg
    path = result_dir / CONTAINER_SETTINGS_PATH.name
    _safe_write_json(path, settings)
    return path


def write_lammps_mcp_config(
    *,
    result_dir: Path,
    blocked_in_filenames: list[str],
    blocked_rst_relpaths: list[str],
    enable_rag: bool = True,
    enable_lammps_validate: bool = False,
) -> Path:
    """Write the MCP config for LAMMPS runs — registers lammps-rag and optionally lammps-validate."""
    servers: dict[str, Any] = {}
    if enable_rag:
        servers["lammps-rag"] = {
            "type": "stdio",
            "command": "uv",
            "args": [
                "run",
                "--script",
                str(CONTAINER_PLUGIN_DIR / "scripts" / "lammps_rag_mcp.py"),
            ],
            "env": {
                "CLAUDE_PLUGIN_ROOT": str(CONTAINER_PLUGIN_DIR),
                "LAMMPS_VECTOR_DB_DIR": str(CONTAINER_LAMMPS_VECTOR_DB_DIR),
                "EXCLUDED_GT_IN_FILENAMES": json.dumps(blocked_in_filenames),
                "EXCLUDED_RST_PATHS": json.dumps(blocked_rst_relpaths),
            },
        }
    if enable_lammps_validate:
        servers["lammps-validate"] = {
            "type": "stdio",
            "command": "uv",
            "args": [
                "run",
                "--script",
                str(CONTAINER_PLUGIN_DIR / "scripts" / "lammps_validate_mcp.py"),
            ],
            "env": {
                "CLAUDE_PLUGIN_ROOT": str(CONTAINER_PLUGIN_DIR),
                # LAMMPS_BINARY forwarded from docker env (binary is installed in the image).
                "LAMMPS_BINARY": os.environ.get("LAMMPS_BINARY", "lammps"),
            },
        }
    mcp_config_path = result_dir / CONTAINER_MCP_CONFIG_PATH.name
    _safe_write_json(mcp_config_path, {"mcpServers": servers})
    return mcp_config_path

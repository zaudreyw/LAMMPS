"""Per-task driver that wires up filtered GEOS, MCP config, retry loop, and runner dispatch."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .contamination import (
    cleanup_filtered_geos_copy,
    create_filtered_geos_copy,
    get_blocked_files_for_task,
)

from .agents import AGENTS
from .claude_settings import (
    _safe_write_json,
    write_claude_mcp_config,
    write_claude_settings,
    write_lammps_claude_settings,
    write_lammps_mcp_config,
)
from .constants import (
    CONTAINER_GEOS_PRIMER_PATH,
    CONTAINER_MCP_CONFIG_PATH,
    CONTAINER_PLUGIN_DIR,
    CONTAINER_VECTOR_DB_DIR,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_GEOS_PRIMER_PATH,
    DEFAULT_LAMMPS_LIB_DIR,
    DEFAULT_PLUGIN_DIR,
    DEFAULT_VECTOR_DB_DIR,
    DOCKER_IMAGE,
    GEOS_LIB_DIR,
    TEMP_GEOS_PARENT,
)
from .docker_cmd import (
    build_claude_native_command,
    build_claude_native_env,
    build_lammps_native_command,
    build_lammps_native_env,
    create_runtime_vector_db_copy,
    preflight_claude_native_mcp,
    preflight_lammps_mcp,
    remove_workspace_geos_primer,
)
from .events import (
    analyze_event_stream_text,
    archive_native_attempt_outputs,
    classify_final_status,
)
from .process_mgr import (
    STOP_REQUESTED,
    _register_process,
    _unregister_process,
    stop_active_processes,
)
from .prompts import (
    build_system_prompt,
    build_task_prompt,
    load_task_instructions,
    native_plugin_prefix,
    no_outputs_retry_prompt,
    pseudo_tool_retry_prompt,
    redact_command_for_display,
)
from .task import run_claude_native_task
from .tool_counts import _new_tool_counts


def run_task(
    task_name: str,
    agent_key: str,
    agents_context: str,
    experiments_dir: Path,
    run_name: str,
    timeout: int,
    dry_run: bool,
    pseudo_tool_retries: int = 1,
    ground_truth_dir: Path | None = None,
    plugin_dir: Path | None = None,
    vector_db_dir: Path | None = None,
    geos_primer_path: Path | None = None,
    claude_model: str | None = None,
    tmp_geos_parent: Path | None = None,
    geos_lib_dir: Path | None = None,
    lammps_lib_dir: Path | None = None,
    extra_blocked_xml_basenames: list[str] | None = None,
    supervisor_spec_dir: Path | None = None,
) -> dict:
    lammps_mode = bool(AGENTS.get(agent_key, {}).get("lammps_mode", False))
    geos_root = (geos_lib_dir if geos_lib_dir is not None else GEOS_LIB_DIR).resolve()
    lammps_root = (lammps_lib_dir if lammps_lib_dir is not None else DEFAULT_LAMMPS_LIB_DIR).resolve()
    agent = AGENTS[agent_key]
    task_dir = experiments_dir / task_name
    result_dir = agent["results_dir"] / run_name / task_name

    # Ensure workspace subdirs exist on the host before mounting
    (result_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (result_dir / "outputs").mkdir(parents=True, exist_ok=True)
    (result_dir / ".claude_home" / ".config").mkdir(parents=True, exist_ok=True)
    (result_dir / ".uv_cache").mkdir(parents=True, exist_ok=True)

    task_instructions = load_task_instructions(task_dir)
    resolved_geos_primer_path = (geos_primer_path or DEFAULT_GEOS_PRIMER_PATH).resolve()
    primer_workspace_path: Path | None = None
    if not lammps_mode:
        remove_workspace_geos_primer(result_dir)
    cheatsheet_path = agent.get("cheatsheet_path")
    # Per-task cheatsheet support (MemP-style retrieval).
    # If `cheatsheet_path_template` is set, format with `{task}` to get
    # a per-task cheatsheet file. Falls back to `cheatsheet_path` if the
    # per-task file doesn't exist.
    cheatsheet_template = agent.get("cheatsheet_path_template")
    if cheatsheet_template is not None:
        per_task_cs = Path(str(cheatsheet_template).format(task=task_name))
        if per_task_cs.exists():
            cheatsheet_path = per_task_cs
        elif cheatsheet_path is None:
            print(f"WARN: cheatsheet template miss for {task_name} at {per_task_cs}")
    cheatsheet_in_workspace = bool(agent.get("cheatsheet_in_workspace", False))
    _plugin_on = bool(agent.get("plugin_enabled", True))
    _rag_on = bool(agent.get("rag_enabled", _plugin_on))
    system_prompt, primer_in_system_prompt = build_system_prompt(
        agents_context,
        resolved_geos_primer_path,
        cheatsheet_path=cheatsheet_path,
        cheatsheet_in_workspace=cheatsheet_in_workspace,
        memory_enabled=bool(agent.get("memory_enabled", False)),
        memory_prompt_hint=bool(agent.get("memory_prompt_hint", True)),
        plugin_enabled=_plugin_on,
        rag_enabled=_rag_on,
        supervisor_enabled=bool(agent.get("supervisor_enabled", False)),
        lammps_mode=lammps_mode,
    )
    # If we're delivering the cheatsheet via the workspace (instead of system prompt),
    # drop the file into the result_dir so Docker bind-mounts it as /workspace/CHEATSHEET.md.
    if cheatsheet_in_workspace and cheatsheet_path is not None and Path(cheatsheet_path).exists():
        ws_cs = result_dir / "CHEATSHEET.md"
        try:
            ws_cs.write_text(Path(cheatsheet_path).read_text())
        except OSError as exc:
            print(f"WARN: could not write workspace cheatsheet: {exc}")
    task_prompt = build_task_prompt(task_instructions)
    prompt = (
        task_prompt
        if agent.get("runner") == "claude_native"
        else f"{system_prompt}\n\n{task_prompt}"
    )
    primer_delivery = "system_prompt" if primer_in_system_prompt else "disabled"

    # Collect blocked files for this experiment.
    blocked_xml_filenames: list[str] = []
    blocked_in_filenames: list[str] = []
    blocked_rst_relpaths: list[str] = []

    if lammps_mode:
        # LAMMPS GT blocking: find *.in files in the per-task GT dir and pass
        # them to the RAG server via EXCLUDED_GT_IN_FILENAMES. No file-system
        # filtering is done — the whole LAMMPS lib is mounted read-only.
        if ground_truth_dir is not None:
            lammps_gt_dir = ground_truth_dir / task_name
            if lammps_gt_dir.exists():
                blocked_in_filenames = [
                    p.name.lower()
                    for p in lammps_gt_dir.iterdir()
                    if p.is_file() and (p.name.lower().endswith(".in") or p.name.lower().startswith("in."))
                ]
        filtered_geos = lammps_root  # not actually mounted as /geos_lib; just a placeholder
        cleanup_filtered_copy = False
    else:
        # GEOS path: variant expansion + RST blocking + filtered hardlink copy.
        if ground_truth_dir is not None:
            blocked = get_blocked_files_for_task(
                task_name,
                ground_truth_dir,
                geos_source_dir=geos_root,
            )
            blocked_xml_filenames = blocked["blocked_xml_filenames"]
            blocked_rst_relpaths = blocked["blocked_rst_paths"]

        if extra_blocked_xml_basenames:
            before = set(blocked_xml_filenames)
            blocked_xml_filenames = sorted(
                set(blocked_xml_filenames) | {b.lower() for b in extra_blocked_xml_basenames}
            )
            added = len(blocked_xml_filenames) - len(before)
            if added:
                print(f"  [extra-blocklist] {added} additional XML basenames blocked for {task_name}")

        cleanup_filtered_copy = False
        if dry_run:
            filtered_geos = geos_root
        else:
            filtered_geos = create_filtered_geos_copy(
                geos_root,
                blocked_xml_basenames=blocked_xml_filenames,
                blocked_rst_relpaths=blocked_rst_relpaths,
                tmp_parent=tmp_geos_parent or TEMP_GEOS_PARENT,
            )
            cleanup_filtered_copy = True

    runner = agent.get("runner", "acpx")
    if runner == "claude_native":
        enable_plugin = agent.get("plugin_enabled", True)
        cleanup_vector_db_copy = False
        runtime_vector_db_dir: Path | None = None
        mcp_config_path: Path | None = None
        if enable_plugin:
            plugin_dir = (plugin_dir or DEFAULT_PLUGIN_DIR).resolve()
            vector_db_dir = (vector_db_dir or DEFAULT_VECTOR_DB_DIR).resolve()
            runtime_vector_db_dir = result_dir / ".vector_db_runtime"
            if not dry_run:
                try:
                    runtime_vector_db_dir = create_runtime_vector_db_copy(vector_db_dir, result_dir)
                    cleanup_vector_db_copy = True
                except Exception:
                    if cleanup_filtered_copy:
                        cleanup_filtered_geos_copy(filtered_geos)
                    raise
            # Supervisor channel for the interactive-autonomy study. The
            # full original spec for this task lives at
            # <supervisor_spec_dir>/<task>/instructions.txt and is mounted
            # read-only at /supervisor/spec.md inside the container. The
            # MCP server reads it; the agent has no reason to look at it.
            supervisor_enabled = bool(agent.get("supervisor_enabled", False))
            supervisor_host_path: Path | None = None
            if supervisor_enabled:
                if supervisor_spec_dir is None:
                    raise ValueError(
                        "supervisor_enabled=True requires --supervisor-spec-dir"
                    )
                supervisor_host_path = (
                    supervisor_spec_dir / task_name / "instructions.txt"
                ).resolve()
                if not supervisor_host_path.exists():
                    raise FileNotFoundError(
                        f"supervisor spec missing: {supervisor_host_path}"
                    )
            hook_enabled = bool(agent.get("stop_hook_enabled", True))
            if lammps_mode:
                mcp_config_path = write_lammps_mcp_config(
                    result_dir=result_dir,
                    blocked_in_filenames=blocked_in_filenames,
                    blocked_rst_relpaths=blocked_rst_relpaths,
                    enable_rag=_rag_on,
                    enable_lammps_validate=bool(agent.get("lammps_validate_mcp_enabled", False)),
                )
                write_lammps_claude_settings(result_dir=result_dir, hook_enabled=hook_enabled)
            else:
                mcp_config_path = write_claude_mcp_config(
                    result_dir=result_dir,
                    blocked_xml_filenames=blocked_xml_filenames,
                    blocked_rst_relpaths=blocked_rst_relpaths,
                    enable_memory=bool(agent.get("memory_enabled", False)),
                    enable_noop=bool(agent.get("noop_mcp_enabled", False)),
                    enable_xmllint=bool(agent.get("xmllint_mcp_enabled", False)),
                    enable_rag=_rag_on,
                    enable_supervisor=supervisor_enabled,
                    supervisor_task_name=task_name,
                    supervisor_model=str(agent.get("supervisor_model", "deepseek-v4-flash")),
                    supervisor_base_url=str(agent.get("supervisor_base_url", "https://api.deepseek.com/v1")),
                    memory_variant=str(agent.get("memory_variant", "lexical")),
                    memory_items_host_path=(
                        Path(agent["memory_items_path"]).resolve()
                        if agent.get("memory_items_path") else None
                    ),
                    memory_embed_index_host_path=(
                        Path(agent["memory_embed_index_path"]).resolve()
                        if agent.get("memory_embed_index_path") else None
                    ),
                )
                write_claude_settings(result_dir=result_dir, hook_enabled=hook_enabled)
        else:
            plugin_dir = None
        native_model = claude_model or agent.get("model") or DEFAULT_CLAUDE_MODEL
        # The native_plugin_prefix is a user-prompt addendum that says
        # "Don't call Skill; use mcp__geos-rag__* tools". Historically gated
        # on plugin_enabled (so a RAG-disabled cell with plugin still
        # loaded gets the prefix — confusing but how prior cells ran).
        # Agents can explicitly opt out via add_native_plugin_prefix=False.
        _add_prefix = bool(agent.get("add_native_plugin_prefix", enable_plugin))
        if _add_prefix:
            native_prompt = f"{native_plugin_prefix(lammps_mode=lammps_mode)}{prompt}"
        else:
            native_prompt = prompt
        if lammps_mode:
            cmd = build_lammps_native_command(
                lammps_lib_dir=lammps_root,
                result_dir=result_dir,
                plugin_dir=plugin_dir,
                vector_db_dir=runtime_vector_db_dir,
                model=native_model,
                system_prompt=system_prompt,
                prompt=native_prompt,
                enable_plugin=enable_plugin,
            )
            docker_env = build_lammps_native_env(
                blocked_in_filenames=blocked_in_filenames,
                blocked_rst_relpaths=blocked_rst_relpaths,
                vector_db_dir=vector_db_dir if enable_plugin else None,
            )
        else:
            cmd = build_claude_native_command(
                filtered_geos=filtered_geos,
                result_dir=result_dir,
                plugin_dir=plugin_dir,
                vector_db_dir=runtime_vector_db_dir,
                model=native_model,
                system_prompt=system_prompt,
                prompt=native_prompt,
                enable_plugin=enable_plugin,
                supervisor_spec_host_path=supervisor_host_path
                if enable_plugin else None,
            )
            docker_env = build_claude_native_env(
                blocked_xml_filenames=blocked_xml_filenames,
                blocked_rst_relpaths=blocked_rst_relpaths,
                vector_db_dir=vector_db_dir if enable_plugin else None,
            )

        if dry_run:
            display = redact_command_for_display(cmd[:-1] + ["<prompt>"])
            print(f"  [DRY RUN] {display}")
            return {"task": task_name, "agent": agent_key, "status": "dry_run"}

        (result_dir / "eval_metadata.json").write_text(
            json.dumps(
                {
                    "task": task_name,
                    "agent": agent_key,
                    "runner": runner,
                    "run_name": run_name,
                    "plugin_enabled": enable_plugin,
                    "plugin_dir": str(plugin_dir) if plugin_dir else None,
                    "plugin_manifest": (
                        str(plugin_dir / ".claude-plugin" / "plugin.json")
                        if plugin_dir else None
                    ),
                    "vector_db_dir": str(vector_db_dir) if enable_plugin else None,
                    "runtime_vector_db_dir": (
                        str(runtime_vector_db_dir) if runtime_vector_db_dir else None
                    ),
                    "container_plugin_dir": str(CONTAINER_PLUGIN_DIR) if enable_plugin else None,
                    "container_vector_db_dir": (
                        str(CONTAINER_VECTOR_DB_DIR) if enable_plugin else None
                    ),
                    "mcp_config_path": str(mcp_config_path) if mcp_config_path else None,
                    "container_mcp_config_path": (
                        str(CONTAINER_MCP_CONFIG_PATH) if enable_plugin else None
                    ),
                    "geos_primer_path": str(resolved_geos_primer_path),
                    "primer_workspace_path": str(primer_workspace_path) if primer_workspace_path else None,
                    "container_geos_primer_path": None,
                    "primer_delivery": primer_delivery,
                    "primer_in_system_prompt": primer_in_system_prompt,
                    "claude_model": native_model,
                    "anthropic_base_url": docker_env.get("ANTHROPIC_BASE_URL"),
                    "blocked_gt_xml_filenames": blocked_xml_filenames,
                    "blocked_rst_relpaths": blocked_rst_relpaths,
                    "filtered_geos_copy": str(filtered_geos),
                    "requires_rag": bool(agent.get("requires_rag")),
                    "started": datetime.now().isoformat(),
                },
                indent=2,
            )
        )

        try:
            _safe_write_json(
                result_dir / "status.json",
                {
                    "task": task_name,
                    "agent": agent_key,
                    "run_name": run_name,
                    "status": "preflight",
                    "process_status": "preflight",
                    "updated": datetime.now().isoformat(),
                    "blocked_gt_xml_filenames": blocked_xml_filenames,
                    **_new_tool_counts(),
                },
            )
            if enable_plugin:
                if lammps_mode:
                    preflight_lammps_mcp(
                        result_dir=result_dir,
                        plugin_dir=plugin_dir,
                        vector_db_dir=runtime_vector_db_dir,
                    )
                else:
                    preflight_claude_native_mcp(
                        result_dir=result_dir,
                        plugin_dir=plugin_dir,
                        vector_db_dir=runtime_vector_db_dir,
                    )

            attempt = 0
            current_cmd = cmd
            while True:
                result = run_claude_native_task(
                    task_name=task_name,
                    agent_key=agent_key,
                    run_name=run_name,
                    cmd=current_cmd,
                    docker_env=docker_env,
                    result_dir=result_dir,
                    timeout=timeout,
                    requires_rag=bool(agent.get("requires_rag")),
                    primer_in_system_prompt=primer_in_system_prompt,
                )
                retryable_status = result.get("status") in {
                    "failed_pseudo_tool",
                    "failed_rag_unavailable",
                    "failed_no_outputs",
                }
                if (
                    not retryable_status
                    or attempt >= pseudo_tool_retries
                    or STOP_REQUESTED.is_set()
                ):
                    return result
                if (
                    result.get("status") in {"failed_pseudo_tool", "failed_rag_unavailable"}
                    and int(result.get("pseudo_tool_calls") or 0) <= 0
                ):
                    return result

                attempt += 1
                archive_native_attempt_outputs(result_dir, attempt)
                if result.get("status") == "failed_no_outputs":
                    notice = no_outputs_retry_prompt(str(result.get("status")))
                else:
                    notice = pseudo_tool_retry_prompt(
                        str(result.get("status")),
                        {
                            "pseudo_tool_counts": result.get("pseudo_tool_counts", {}),
                        },
                    )
                retry_prompt = f"{native_prompt}{notice}"
                if lammps_mode:
                    current_cmd = build_lammps_native_command(
                        lammps_lib_dir=lammps_root,
                        result_dir=result_dir,
                        plugin_dir=plugin_dir,
                        vector_db_dir=runtime_vector_db_dir,
                        model=native_model,
                        system_prompt=system_prompt,
                        prompt=retry_prompt,
                        enable_plugin=enable_plugin,
                    )
                else:
                    current_cmd = build_claude_native_command(
                        filtered_geos=filtered_geos,
                        result_dir=result_dir,
                        plugin_dir=plugin_dir,
                        vector_db_dir=runtime_vector_db_dir,
                        model=native_model,
                        system_prompt=system_prompt,
                        prompt=retry_prompt,
                        enable_plugin=enable_plugin,
                        supervisor_spec_host_path=supervisor_host_path
                        if enable_plugin else None,
                    )
                _safe_write_json(
                    result_dir / "status.json",
                    {
                        "task": task_name,
                        "agent": agent_key,
                        "run_name": run_name,
                        "status": "retrying_agent_output",
                        "process_status": "retrying_agent_output",
                        "updated": datetime.now().isoformat(),
                        "retry_attempt": attempt,
                        "previous_status": result.get("status"),
                        "pseudo_tool_counts": result.get("pseudo_tool_counts", {}),
                        "blocked_gt_xml_filenames": blocked_xml_filenames,
                        **_new_tool_counts(),
                    },
                )
        except Exception as exc:
            (result_dir / "exit_code.txt").write_text("error")
            (result_dir / "stderr.txt").write_text(str(exc))
            _safe_write_json(
                result_dir / "status.json",
                {
                    "task": task_name,
                    "agent": agent_key,
                    "run_name": run_name,
                    "status": "error",
                    "process_status": "error",
                    "error": str(exc),
                    "updated": datetime.now().isoformat(),
                },
            )
            return {"task": task_name, "agent": agent_key, "status": "error", "error": str(exc)}
        finally:
            if cleanup_vector_db_copy:
                shutil.rmtree(runtime_vector_db_dir, ignore_errors=True)
            if cleanup_filtered_copy:
                cleanup_filtered_geos_copy(filtered_geos)

    model = agent.get("model")
    api_key = os.environ.get(agent["api_key_env"], "")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")

    # For cursor: prepend /model slash command as the most reliable way to set
    # the model within an acpx session (Cursor ACP doesn't advertise models yet).
    if model and agent["acpx_name"] == "cursor":
        prompt = f"/model {model}\n\n{prompt}"

    extra_env: list[str] = []
    if model and agent["acpx_name"] == "cursor":
        extra_env += ["-e", f"CURSOR_MODEL={model}"]

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{filtered_geos}:/geos_lib:ro",
        "-v", f"{result_dir}:/workspace:rw",
        "-e", f"{agent['api_key_env']}={api_key}",
        "-e", f"ANTHROPIC_API_KEY={anthropic_key}",
        *extra_env,
        DOCKER_IMAGE,
        "acpx",
        "--approve-reads",
        "--format", "json",
        "--cwd", "/workspace",
        agent["acpx_name"],
        "exec", prompt,
    ]

    if dry_run:
        display = redact_command_for_display(cmd[:-1] + ["<prompt>"])
        print(f"  [DRY RUN] {display}")
        return {"task": task_name, "agent": agent_key, "status": "dry_run"}

    started_time = time.time()
    started_iso = datetime.now().isoformat()

    def write_acpx_status(
        status: str,
        *,
        process_status: str | None = None,
        exit_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        error: str | None = None,
    ) -> None:
        analysis = analyze_event_stream_text(stdout) if stdout else {
            "counts": _new_tool_counts(),
            "latest_stdout": [],
            "latest_agent_response": "",
        }
        payload: dict[str, Any] = {
            "task": task_name,
            "agent": agent_key,
            "run_name": run_name,
            "status": status,
            "process_status": process_status or status,
            "exit_code": exit_code,
            "started": started_iso,
            "updated": datetime.now().isoformat(),
            "elapsed_seconds": round(time.time() - started_time, 1),
            "latest_stdout": analysis["latest_stdout"] or stdout.splitlines()[-40:],
            "latest_agent_response": analysis["latest_agent_response"],
            "latest_stderr": stderr.splitlines()[-40:],
            **analysis["counts"],
        }
        if error:
            payload["error"] = error
        _safe_write_json(result_dir / "status.json", payload)

    write_acpx_status("running", process_status="running")
    heartbeat_stop = threading.Event()
    proc_holder: dict[str, subprocess.Popen[str] | None] = {"proc": None}

    def heartbeat_acpx_status() -> None:
        while not heartbeat_stop.wait(2.0):
            proc = proc_holder["proc"]
            if proc is None:
                continue
            if proc.poll() is not None:
                break
            write_acpx_status("running", process_status="running")

    heartbeat_thread = threading.Thread(target=heartbeat_acpx_status, daemon=True)
    heartbeat_thread.start()

    # Write a metadata file so the run config is auditable
    (result_dir / "eval_metadata.json").write_text(
        json.dumps(
            {
                "task": task_name,
                "agent": agent_key,
                "run_name": run_name,
                "blocked_gt_xml_filenames": blocked_xml_filenames,
                "filtered_geos_copy": str(filtered_geos),
                "geos_primer_path": str(resolved_geos_primer_path),
                "primer_workspace_path": str(primer_workspace_path) if primer_workspace_path else None,
                "container_geos_primer_path": None,
                "primer_delivery": primer_delivery,
                "primer_in_system_prompt": primer_in_system_prompt,
                "started": started_iso,
            },
            indent=2,
        )
    )

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        proc_holder["proc"] = proc
        _register_process(proc)
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        finally:
            _unregister_process(proc)
        if STOP_REQUESTED.is_set() and proc.returncode != 0:
            status = "interrupted"
        else:
            status = "success" if proc.returncode == 0 else "failed"
        if status == "success":
            status = classify_final_status(
                process_status=status,
                requires_rag=False,
                counts=analyze_event_stream_text(stdout)["counts"],
            )
        (result_dir / "acpx_output.json").write_text(stdout)
        (result_dir / "stderr.txt").write_text(stderr)
        (result_dir / "exit_code.txt").write_text(str(proc.returncode))
        write_acpx_status(
            status,
            process_status=status,
            exit_code=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )

        return {
            "task": task_name,
            "agent": agent_key,
            "status": status,
            "exit_code": proc.returncode,
        }

    except KeyboardInterrupt:
        stop_active_processes()
        (result_dir / "exit_code.txt").write_text("interrupted")
        (result_dir / "stderr.txt").write_text("Interrupted by user")
        write_acpx_status(
            "interrupted",
            process_status="interrupted",
            stderr="Interrupted by user",
        )
        return {"task": task_name, "agent": agent_key, "status": "interrupted"}

    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        _unregister_process(proc)
        (result_dir / "acpx_output.json").write_text(stdout)
        (result_dir / "stderr.txt").write_text(stderr)
        (result_dir / "exit_code.txt").write_text("timeout")
        write_acpx_status(
            "timeout",
            process_status="timeout",
            stdout=stdout,
            stderr=stderr,
        )
        return {"task": task_name, "agent": agent_key, "status": "timeout"}

    except Exception as exc:
        (result_dir / "exit_code.txt").write_text("error")
        (result_dir / "stderr.txt").write_text(str(exc))
        write_acpx_status(
            "error",
            process_status="error",
            stderr=str(exc),
            error=str(exc),
        )
        return {"task": task_name, "agent": agent_key, "status": "error", "error": str(exc)}

    finally:
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=5)
        if cleanup_filtered_copy:
            cleanup_filtered_geos_copy(filtered_geos)

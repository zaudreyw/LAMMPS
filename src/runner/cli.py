"""
Docker-based eval harness for comparing claude_code and cursor_composer2 agents
on GEOS XML authoring tasks via acpx.

Usage:
    # Run all tasks for both agents under experiment_run1
    python run_eval.py --run experiment_run1

    # Specific agents only
    python run_eval.py --run experiment_run1 --agents claude_code

    # Include only specific tasks
    python run_eval.py --run experiment_run1 --include TutorialDeadOilEgg ExampleEDPWellbore

    # Exclude specific tasks
    python run_eval.py --run experiment_run1 --exclude TutorialDeadOilEgg

    # Override the experiments source directory
    python run_eval.py --run experiment_run1 --experiments-dir /path/to/my/tasks

    # Dry run (prints docker commands without executing)
    python run_eval.py --run experiment_run1 --dry-run

    # Adjust concurrency and timeout
    python run_eval.py --run experiment_run1 --workers 4 --timeout 900

Build the Docker image first:
    docker build -t geos-eval run/

Expected layout after a run:
    /home/brianliu/data/eval/
    ├── claude_code/
    │   └── experiment_run1/
    │       └── <task>/
    │           ├── inputs/          ← agent-generated XML files
    │           ├── outputs/         ← agent-generated outputs
    │           ├── acpx_output.json ← stdout from acpx
    │           ├── stderr.txt       ← stderr from acpx
    │           └── exit_code.txt    ← process exit code
    └── cursor_composer2/
        └── experiment_run1/
            └── <task>/
                └── ...

To evaluate results afterwards, use batch_lxml_evaluate.py:
    uv run python scripts/eval/batch_lxml_evaluate.py \\
        --experiments-dir /home/brianliu/data/eval/claude_code/experiment_run1 \\
        --ground-truth-dir /home/brianliu/data/eval/experiments_gt \\
        --results-dir /home/brianliu/data/eval/claude_code_results/experiment_run1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

# Importing the package runs env-var aliasing; importing this module too.
from . import _env_bootstrap  # noqa: F401

from .contamination import get_blocked_files_for_task

from .agents import AGENTS
from .constants import (
    CONTAINER_GEOS_PRIMER_PATH,
    DATA_DIR,
    DEFAULT_CLAUDE_MODEL,
    DEFAULT_GEOS_LIB_DIR,
    DEFAULT_GEOS_PRIMER_PATH,
    DEFAULT_LAMMPS_LIB_DIR,
    DEFAULT_LAMMPS_VECTOR_DB_DIR,
    DEFAULT_PLUGIN_DIR,
    DEFAULT_TIMEOUT,
    DEFAULT_VECTOR_DB_DIR,
    EXPERIMENTS_DIR,
    GROUND_TRUTH_DIR,
    LAMMPS_EXPERIMENTS_DIR,
    TEMP_GEOS_PARENT,
)
from .dashboard.server import start_dashboard_server
from .orchestrator import run_task
from .process_mgr import stop_active_processes
from .prompts import load_agents_md
from .run_lock import RunLockHeld, acquire_run_lock
from .signal_logger import install_signal_logger


class C:
    GREEN   = "\033[92m"
    WARNING = "\033[93m"
    FAIL    = "\033[91m"
    CYAN    = "\033[96m"
    BOLD    = "\033[1m"
    HEADER  = "\033[95m"
    ENDC    = "\033[0m"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GEOS eval harness: runs claude_code and cursor_composer2 via Docker + acpx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run", "-r",
        required=True,
        metavar="RUN_NAME",
        help="Name of the experiment run subfolder, e.g. experiment_run1. "
             "Results land at <agent_dir>/<run>/<task>/",
    )
    parser.add_argument(
        "--experiments-dir", "-d",
        type=Path,
        default=EXPERIMENTS_DIR,
        help=f"Directory containing task subdirs with instructions.txt "
             f"(default: {EXPERIMENTS_DIR})",
    )
    parser.add_argument(
        "--supervisor-spec-dir",
        type=Path,
        default=None,
        help="Directory containing per-task FULL original specs for the "
             "simulated-supervisor MCP server. Required when running an "
             "agent variant with supervisor_enabled=True.",
    )
    parser.add_argument(
        "--agents", "-a",
        nargs="+",
        choices=list(AGENTS.keys()),
        default=list(AGENTS.keys()),
        help="Agents to evaluate (default: all)",
    )
    parser.add_argument(
        "--include", "-i",
        nargs="+",
        metavar="TASK_NAME",
        help="Run only these tasks (default: all tasks in experiments dir)",
    )
    parser.add_argument(
        "--exclude", "-x",
        nargs="+",
        metavar="TASK_NAME",
        default=[],
        help="Skip these tasks (applied after --include)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per task in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--pseudo-tool-retries",
        type=int,
        default=1,
        help="Retries per task when an agent prints non-executed pseudo tool invocations (default: 1)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=2,
        help="Max concurrent docker runs (default: 2; keep low to avoid OOM)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print docker commands without executing",
    )
    parser.add_argument(
        "--results-root-dir",
        type=Path,
        default=DATA_DIR / "eval",
        help=f"Root directory for per-agent result folders "
             f"(default: {DATA_DIR / 'eval'})",
    )
    parser.add_argument(
        "--tmp-geos-parent",
        type=Path,
        default=TEMP_GEOS_PARENT,
        help=f"Directory for per-task filtered GEOS copies "
             f"(default: {TEMP_GEOS_PARENT})",
    )
    parser.add_argument(
        "--plugin-dir",
        type=Path,
        default=DEFAULT_PLUGIN_DIR,
        help=f"Claude Code plugin directory for claude_code_repo3_plugin "
             f"(default: {DEFAULT_PLUGIN_DIR})",
    )
    parser.add_argument(
        "--vector-db-dir",
        type=Path,
        default=DEFAULT_VECTOR_DB_DIR,
        help=f"Host GEOS vector DB directory mounted for repo3 RAG "
             f"(default: {DEFAULT_VECTOR_DB_DIR})",
    )
    parser.add_argument(
        "--geos-primer-path",
        type=Path,
        default=DEFAULT_GEOS_PRIMER_PATH,
        help=f"GEOS primer markdown inlined into the agent system prompt; "
             f"{CONTAINER_GEOS_PRIMER_PATH} is not created in task workspaces "
             f"(default: {DEFAULT_GEOS_PRIMER_PATH}). "
             f"NOTE: only takes effect when AGENTS.md does NOT already contain "
             f"a `# GEOS Primer` section, OR when --strip-baked-primer is set.",
    )
    parser.add_argument(
        "--agents-md-path",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to the AGENTS.md system-prompt file. "
             "Defaults to run/AGENTS.md. Pass run/AGENTS_lammps.md to use "
             "the LAMMPS harness system prompt.",
    )
    parser.add_argument(
        "--strip-baked-primer",
        action="store_true",
        default=False,
        help="Strip the embedded primer section (# GEOS Primer or # LAMMPS Primer) "
             "out of AGENTS.md before injecting it as system context, so the file "
             "passed via --geos-primer-path is actually inlined. Use this for any "
             "primer ablation.",
    )
    parser.add_argument(
        "--claude-model",
        default=DEFAULT_CLAUDE_MODEL,
        help=f"Model passed to Claude Code for native Claude runs "
             f"(default: {DEFAULT_CLAUDE_MODEL})",
    )
    parser.add_argument(
        "--extend-blocklist-with-test",
        action="store_true",
        default=False,
        help="Extend each task's blocklist with the union of all 17 test-task "
             "blocked_gt_xml_filenames. Required when harvesting trajectories from "
             "training tasks that will feed memory artifacts used at test time. "
             "Loads the blocklist from misc/memory_artifacts/test_blocklist.json.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Serve a browser dashboard for live task status and output",
    )
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Serve the dashboard for the selected run/agents/tasks without launching experiments",
    )
    parser.add_argument(
        "--dashboard-host",
        default="127.0.0.1",
        help="Dashboard bind host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8765,
        help="Dashboard port; if busy, the next available port is used (default: 8765)",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        default=GROUND_TRUTH_DIR,
        metavar="DIR",
        help=f"Directory containing per-task ground-truth subdirs whose XML filenames "
             f"will be blocked from the agent. Pass an empty string to disable. "
             f"(default: {GROUND_TRUTH_DIR})",
    )
    parser.add_argument(
        "--geos-lib-dir",
        type=Path,
        default=DEFAULT_GEOS_LIB_DIR,
        metavar="DIR",
        help=f"GEOS source tree; filtered copies are mounted at /geos_lib in Docker "
             f"(default: {DEFAULT_GEOS_LIB_DIR})",
    )
    parser.add_argument(
        "--lammps-lib-dir",
        type=Path,
        default=DEFAULT_LAMMPS_LIB_DIR,
        metavar="DIR",
        help=f"LAMMPS source tree mounted read-only at /lammps_lib in Docker. "
             f"Only used when running lammps_mode agents. "
             f"(default: {DEFAULT_LAMMPS_LIB_DIR})",
    )
    parser.add_argument(
        "--force-unlock",
        action="store_true",
        help="Override the per-run PID lock if a stale or zombie lockfile is "
             "blocking start. Only use when you are certain no other harness "
             "process is running this --run name.",
    )
    args = parser.parse_args()

    for agent_key, agent in AGENTS.items():
        agent["results_dir"] = args.results_root_dir / agent_key

    geos_lib_resolved = args.geos_lib_dir.resolve()

    selected_native_claude = any(
        AGENTS[agent_key].get("runner") == "claude_native"
        for agent_key in args.agents
    )
    selected_plugin_agent = any(
        AGENTS[agent_key].get("runner") == "claude_native"
        and AGENTS[agent_key].get("plugin_enabled", True)
        for agent_key in args.agents
    )
    if selected_plugin_agent and not args.dashboard_only:
        if not args.plugin_dir.exists():
            print(f"{C.FAIL}Error: plugin dir not found: {args.plugin_dir}{C.ENDC}")
            sys.exit(1)
        if not (args.plugin_dir / ".claude-plugin" / "plugin.json").exists():
            print(
                f"{C.FAIL}Error: plugin manifest not found under: "
                f"{args.plugin_dir / '.claude-plugin' / 'plugin.json'}{C.ENDC}"
            )
            sys.exit(1)
        if not args.vector_db_dir.exists():
            print(f"{C.FAIL}Error: vector DB dir not found: {args.vector_db_dir}{C.ENDC}")
            sys.exit(1)
    if selected_native_claude and not args.dashboard_only:
        if not args.dry_run and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            print(
                f"{C.FAIL}Error: ANTHROPIC_AUTH_TOKEN is required for "
                f"claude_native agents. Export it in your shell before running.{C.ENDC}"
            )
            sys.exit(1)

    selected_lammps_agents = [
        k for k in args.agents if AGENTS[k].get("lammps_mode")
    ]
    selected_geos_agents = [
        k for k in args.agents if not AGENTS[k].get("lammps_mode")
    ]
    if not args.dry_run and not args.dashboard_only:
        if selected_geos_agents and not geos_lib_resolved.is_dir():
            print(
                f"{C.FAIL}Error: GEOS source dir not found: {geos_lib_resolved}. "
                f"Set --geos-lib-dir to your GEOS checkout.{C.ENDC}"
            )
            sys.exit(1)
        if selected_lammps_agents:
            lammps_lib_resolved = args.lammps_lib_dir.resolve()
            if not lammps_lib_resolved.is_dir():
                print(
                    f"{C.FAIL}Error: LAMMPS source dir not found: {lammps_lib_resolved}. "
                    f"Set --lammps-lib-dir to your LAMMPS checkout.{C.ENDC}"
                )
                sys.exit(1)

    if not args.dashboard_only and not args.geos_primer_path.exists():
        print(
            f"{C.WARNING}Warning: GEOS primer not found: {args.geos_primer_path}; "
            f"primer injection disabled.{C.ENDC}"
        )

    # Normalise ground-truth-dir: treat missing or empty-string as None
    ground_truth_dir: Path | None = args.ground_truth_dir
    if ground_truth_dir is not None and not ground_truth_dir.exists():
        print(
            f"{C.WARNING}Warning: --ground-truth-dir '{ground_truth_dir}' does not exist; "
            f"GT XML blocking disabled.{C.ENDC}"
        )
        ground_truth_dir = None

    experiments_dir: Path = args.experiments_dir

    # Validate experiments directory
    if not experiments_dir.exists():
        print(f"{C.FAIL}Error: experiments dir not found: {experiments_dir}{C.ENDC}")
        sys.exit(1)

    # Discover all tasks
    all_tasks = sorted(d.name for d in experiments_dir.iterdir() if d.is_dir())

    # Apply --include filter
    if args.include:
        missing = [t for t in args.include if t not in all_tasks]
        if missing:
            print(f"{C.WARNING}Warning: tasks not found in {experiments_dir}: {missing}{C.ENDC}")
        tasks = [t for t in args.include if t in all_tasks]
    else:
        tasks = all_tasks

    # Apply --exclude filter
    if args.exclude:
        excluded = set(args.exclude)
        tasks = [t for t in tasks if t not in excluded]

    if not tasks:
        print(f"{C.FAIL}No tasks to run.{C.ENDC}")
        sys.exit(1)

    blocked_gt_by_task: dict[str, list[str]] = {}
    if ground_truth_dir is not None:
        blocked_gt_by_task = {
            task: get_blocked_files_for_task(
                task, ground_truth_dir, geos_source_dir=geos_lib_resolved,
            )["blocked_xml_filenames"]
            for task in tasks
        }

    agents_context = load_agents_md(
        strip_baked_primer=args.strip_baked_primer,
        agents_md_path=args.agents_md_path,
    )
    combos = [(task, agent) for task in tasks for agent in args.agents]

    # Show where results will land
    result_paths = {
        agent_key: AGENTS[agent_key]["results_dir"] / args.run
        for agent_key in args.agents
    }

    print(f"\n{C.BOLD}{C.HEADER}{'=' * 70}{C.ENDC}")
    print(f"{C.BOLD}{C.HEADER}  GEOS Eval Harness{C.ENDC}")
    print(f"{C.BOLD}{C.HEADER}{'=' * 70}{C.ENDC}")
    print(f"  Run name       : {args.run}")
    print(f"  Experiments dir: {experiments_dir}")
    print(f"  Tasks          : {len(tasks)}")
    print(f"  Agents         : {args.agents}")
    print(f"  Combos         : {len(combos)}")
    print(f"  Timeout        : {args.timeout}s per task")
    print(f"  Workers        : {args.workers}")
    print(f"  Pseudo retries : {args.pseudo_tool_retries}")
    print(f"  Dry run        : {args.dry_run}")
    print(f"  GT XML blocking: {ground_truth_dir or 'disabled'}")
    print(f"  Results root   : {args.results_root_dir}")
    print(f"  Temp GEOS dir  : {args.tmp_geos_parent}")
    print(f"  GEOS lib dir   : {geos_lib_resolved}")
    print(f"  GEOS primer    : {args.geos_primer_path if args.geos_primer_path.exists() else 'disabled'}")
    if selected_native_claude:
        print(f"  Plugin dir     : {args.plugin_dir}")
        print(f"  Vector DB dir  : {args.vector_db_dir}")
        print(f"  Claude model   : {args.claude_model}")
    for agent_key, path in result_paths.items():
        print(f"  Results ({agent_key}): {path}")
    print(f"  Started        : {datetime.now().isoformat()}")
    print(f"{C.BOLD}{C.HEADER}{'=' * 70}{C.ENDC}\n")

    dashboard_server: ThreadingHTTPServer | None = None
    if args.dashboard or args.dashboard_only:
        try:
            dashboard_server, dashboard_url = start_dashboard_server(
                run_name=args.run,
                agent_keys=args.agents,
                task_names=tasks,
                blocked_gt_by_task=blocked_gt_by_task,
                host=args.dashboard_host,
                port=args.dashboard_port,
            )
            print(f"{C.CYAN}Dashboard:{C.ENDC} {dashboard_url}\n")
        except Exception as exc:
            print(f"{C.FAIL}Error: failed to start dashboard: {exc}{C.ENDC}")
            sys.exit(1)

    if args.dashboard_only:
        assert dashboard_server is not None
        print(f"{C.CYAN}Dashboard-only mode:{C.ENDC} no experiments will be launched.")
        print("Press Ctrl-C to stop the dashboard server.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\nStopping dashboard server.")
            dashboard_server.shutdown()
        return

    results: list[dict] = []

    # Load the test-blocklist union if --extend-blocklist-with-test is set.
    # See RN-003 P2 #7: harvesting trajectories from training tasks that will feed
    # memory artifacts used at test time requires hiding ALL test-task GT basenames
    # from every training run, not just the per-task defaults.
    extra_blocked_xml_basenames: list[str] = []
    if getattr(args, "extend_blocklist_with_test", False):
        test_bl_path = Path("/home/matt/sci/repo3/misc/memory_artifacts/test_blocklist.json")
        if not test_bl_path.exists():
            print(f"{C.FAIL}ERROR: --extend-blocklist-with-test requires {test_bl_path} "
                  f"(build via scripts/memory/compute_test_blocklist.py){C.ENDC}")
            sys.exit(1)
        bl_data = json.loads(test_bl_path.read_text())
        extra_blocked_xml_basenames = list(bl_data.get("union_xml", []))
        print(f"{C.CYAN}--extend-blocklist-with-test:{C.ENDC} "
              f"unioning {len(extra_blocked_xml_basenames)} test-task blocked basenames into every task's blocklist")

    # Lock + signal logger: prevent two harness invocations from sharing a
    # --run name and record any SIGINT/SIGTERM with sender context. Skip for
    # dry runs and dashboard-only invocations (neither launches experiments).
    use_run_guards = not (args.dry_run or args.dashboard_only)
    lock_cm = None
    if use_run_guards:
        signal_log_path = (
            args.results_root_dir / ".run_signals" / f"{args.run}.jsonl"
        )
        install_signal_logger(signal_log_path)
        lock_cm = acquire_run_lock(
            args.results_root_dir,
            args.run,
            command=sys.argv,
            force=args.force_unlock,
        )
        try:
            lock_cm.__enter__()
        except RunLockHeld as exc:
            print(f"{C.FAIL}Error: {exc}{C.ENDC}")
            sys.exit(2)

    executor = ThreadPoolExecutor(max_workers=args.workers)
    futures = {
        executor.submit(
            run_task,
            task_name=task,
            agent_key=agent,
            agents_context=agents_context,
            experiments_dir=experiments_dir,
            run_name=args.run,
            timeout=args.timeout,
            dry_run=args.dry_run,
            pseudo_tool_retries=args.pseudo_tool_retries,
            ground_truth_dir=ground_truth_dir,
            plugin_dir=args.plugin_dir,
            vector_db_dir=args.vector_db_dir,
            geos_primer_path=args.geos_primer_path,
            claude_model=args.claude_model,
            tmp_geos_parent=args.tmp_geos_parent,
            geos_lib_dir=geos_lib_resolved,
            lammps_lib_dir=args.lammps_lib_dir.resolve(),
            extra_blocked_xml_basenames=extra_blocked_xml_basenames,
            supervisor_spec_dir=args.supervisor_spec_dir,
        ): (task, agent)
        for task, agent in combos
    }
    seen_futures: set[Any] = set()
    try:
        for i, future in enumerate(as_completed(futures), 1):
            seen_futures.add(future)
            task, agent = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = result.get("status", "?")
                color = C.GREEN if status == "success" else (C.WARNING if status == "dry_run" else C.FAIL)
                print(f"[{i:3d}/{len(combos)}] {color}{status:<14}{C.ENDC}  {agent:<28}  {task}")
            except Exception as exc:
                results.append({"task": task, "agent": agent, "status": "error", "error": str(exc)})
                print(f"[{i:3d}/{len(combos)}] {C.FAIL}ERROR         {C.ENDC}  {agent:<28}  {task}  ({exc})")
    except KeyboardInterrupt:
        print(f"\n{C.WARNING}Interrupt received. Stopping running agents; dashboard will stay up.{C.ENDC}")
        stop_active_processes()
        for future, (task, agent) in futures.items():
            if future.cancel():
                results.append({"task": task, "agent": agent, "status": "cancelled"})
        executor.shutdown(wait=True, cancel_futures=True)
        for future, (task, agent) in futures.items():
            if future in seen_futures or future.cancelled():
                continue
            if future.done():
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append({"task": task, "agent": agent, "status": "error", "error": str(exc)})
            else:
                results.append({"task": task, "agent": agent, "status": "interrupted"})
    else:
        executor.shutdown(wait=True)

    # Summary
    succeeded = sum(1 for r in results if r["status"] == "success")
    failed    = sum(1 for r in results if r["status"] not in ("success", "dry_run"))
    print(f"\n{C.BOLD}Done{C.ENDC}: {C.GREEN}{succeeded} succeeded{C.ENDC}, "
          f"{C.FAIL}{failed} failed{C.ENDC} / {len(combos)} total")

    if failed:
        print(f"\n{C.FAIL}Failed tasks:{C.ENDC}")
        for r in results:
            if r["status"] not in ("success", "dry_run"):
                error_text = f": {r.get('error', '')}" if r.get("error") else ""
                print(f"  [{r['status']}] {r['agent']} / {r['task']}{error_text}")

    if dashboard_server is not None:
        print(f"\n{C.CYAN}Dashboard is still running:{C.ENDC} {dashboard_url}")
        print("Press Ctrl-C to stop the dashboard server.")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\nStopping dashboard server.")
            dashboard_server.shutdown()

    if lock_cm is not None:
        try:
            lock_cm.__exit__(None, None, None)
        except Exception:
            pass

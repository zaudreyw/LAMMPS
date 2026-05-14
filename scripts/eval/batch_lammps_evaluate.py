#!/usr/bin/env python3
"""Evaluate LAMMPS agent outputs against ground truth.

Two-stage evaluation for each task:

  Stage 1 — Structural checks
    Parses both the agent .in file and the GT .in file to verify key
    commands and numeric parameters are present and match within tolerance.

  Stage 2 — LLM judge
    Sends the task spec, GT script, and agent script to an LLM and asks
    for a 0-10 score with a brief rationale.

Usage:
    # Evaluate one run against GT
    uv run python scripts/eval/batch_lammps_evaluate.py \\
        --agent-run-dir data/eval/lammps_plugin/lammps_run1 \\
        --ground-truth-dir data/eval/experiments_lammps_gt \\
        --experiments-dir data/eval/experiments_lammps \\
        --results-dir data/eval/lammps_scores/lammps_run1

    # Skip LLM judge, structural checks only
    uv run python scripts/eval/batch_lammps_evaluate.py \\
        --agent-run-dir data/eval/lammps_plugin/lammps_run1 \\
        --ground-truth-dir data/eval/experiments_lammps_gt \\
        --experiments-dir data/eval/experiments_lammps \\
        --results-dir data/eval/lammps_scores/lammps_run1 \\
        --no-llm-judge
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Structural parser
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    """Remove inline and full-line LAMMPS comments."""
    lines = []
    for line in text.splitlines():
        line = line.split("#")[0].rstrip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def parse_in_file(path: Path) -> dict[str, Any]:
    """Extract key fields from a LAMMPS .in script into a flat dict."""
    if not path.exists():
        return {}
    text = _strip_comments(path.read_text())
    result: dict[str, Any] = {}

    def _first(pattern: str, flags: int = re.IGNORECASE) -> str | None:
        m = re.search(pattern, text, flags)
        return m.group(1).strip() if m else None

    def _all(pattern: str, flags: int = re.IGNORECASE) -> list[str]:
        return [m.group(1).strip() for m in re.finditer(pattern, text, flags)]

    result["units"]      = _first(r"^\s*units\s+(\S+)", re.M | re.I)
    result["atom_style"] = _first(r"^\s*atom_style\s+(\S+)", re.M | re.I)
    result["dimension"]  = _first(r"^\s*dimension\s+(\d+)", re.M | re.I)
    result["pair_style"] = _first(r"^\s*pair_style\s+(.+)", re.M | re.I)
    result["bond_style"] = _first(r"^\s*bond_style\s+(\S+)", re.M | re.I)
    result["angle_style"]= _first(r"^\s*angle_style\s+(\S+)", re.M | re.I)
    result["kspace"]     = _first(r"^\s*kspace_style\s+(.+)", re.M | re.I)
    result["timestep"]   = _first(r"^\s*timestep\s+([\d.eE+-]+)", re.M | re.I)
    result["run_steps"]  = _all(r"^\s*run\s+(\d+)", re.M | re.I)
    result["minimize"]   = bool(re.search(r"^\s*minimize\b", text, re.M | re.I))
    result["thermo"]     = _first(r"^\s*thermo\s+(\d+)", re.M | re.I)
    result["fixes"]      = _all(r"^\s*fix\s+\S+\s+\S+\s+(\S+)", re.M | re.I)
    result["dumps"]      = _all(r"^\s*dump\s+\S+\s+\S+\s+\S+\s+(\d+)", re.M | re.I)

    # Temperature from nvt/npt
    temp_m = re.search(r"\b(?:nvt|npt)\b.*temp\s+([\d.]+)\s+([\d.]+)", text, re.I)
    result["temp_target"] = float(temp_m.group(2)) if temp_m else None

    # velocity create temperature
    vel_m = re.search(r"^\s*velocity\s+\S+\s+create\s+([\d.]+)", text, re.M | re.I)
    result["vel_temp"] = float(vel_m.group(1)) if vel_m else None

    return result


# ---------------------------------------------------------------------------
# Structural check logic
# ---------------------------------------------------------------------------

TASK_CHECKS: dict[str, list[tuple[str, Any, str]]] = {
    "lj_melt": [
        ("units",       "lj",      "units must be lj"),
        ("atom_style",  "atomic",  "atom_style must be atomic"),
    ],
    "nvt_water": [
        ("units",       "real",    "units must be real"),
        ("atom_style",  "full",    "atom_style must be full"),
        ("bond_style",  "harmonic","bond_style must be harmonic"),
        ("angle_style", "harmonic","angle_style must be harmonic"),
    ],
    "crack_2d": [
        ("units",       "lj",      "units must be lj"),
        ("atom_style",  "atomic",  "atom_style must be atomic"),
        ("dimension",   "2",       "dimension must be 2"),
    ],
}

REQUIRED_COMMANDS: dict[str, list[str]] = {
    "lj_melt":    ["units", "atom_style", "run_steps"],
    "nvt_water":  ["units", "atom_style", "bond_style", "angle_style", "kspace", "run_steps"],
    "crack_2d":   ["units", "atom_style", "dimension", "run_steps"],
}


def structural_check(task: str, parsed: dict[str, Any]) -> dict[str, Any]:
    """Run structural checks, return pass/fail per check with detail."""
    results: list[dict] = []
    passed = 0

    # Required command presence
    for field in REQUIRED_COMMANDS.get(task, []):
        val = parsed.get(field)
        ok = bool(val)
        results.append({"check": f"has_{field}", "passed": ok,
                         "detail": str(val) if ok else "MISSING"})
        if ok:
            passed += 1

    # Value checks
    for field, expected, desc in TASK_CHECKS.get(task, []):
        actual = (parsed.get(field) or "").lower()
        ok = expected.lower() in actual
        results.append({"check": desc, "passed": ok,
                         "detail": f"got '{actual}'"})
        if ok:
            passed += 1

    # Has at least one fix with nve/nvt/npt
    integrators = [f for f in parsed.get("fixes", [])
                   if f.lower() in ("nve", "nvt", "npt", "langevin")]
    has_integrator = bool(integrators)
    results.append({"check": "has_integrator_fix",
                    "passed": has_integrator,
                    "detail": str(integrators) if has_integrator else "MISSING nve/nvt/npt fix"})
    if has_integrator:
        passed += 1

    total = len(results)
    return {"score": round(passed / total, 3) if total else 0.0,
            "passed": passed, "total": total, "checks": results}


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """\
You are evaluating a LAMMPS molecular-dynamics input script written by an AI agent.

## Task specification
{spec}

## Reference (ground truth) script
```lammps
{gt_script}
```

## Agent-generated script
```lammps
{agent_script}
```

Score the agent script on a scale of 0–10 where:
  10 = Correct physics, correct parameters, clean syntax, complete
   7 = Minor parameter deviations (e.g. wrong seed, slightly off cutoff)
   4 = Correct physics approach but missing key components or wrong ensemble
   1 = Wrong units, wrong atom_style, or fundamentally incorrect physics
   0 = No valid LAMMPS script produced

Respond with ONLY valid JSON in this exact format:
{{
  "score": <integer 0-10>,
  "rationale": "<one or two sentences>",
  "key_issues": ["<issue 1>", "<issue 2>"]
}}
"""


def llm_judge(
    task: str,
    spec: str,
    gt_script: str,
    agent_script: str,
    model: str = "anthropic/claude-sonnet-4-6",
) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai package not installed", "score": None}

    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        return {"error": "no API key found", "score": None}

    base_url = os.environ.get("OPENAI_BASE_URL",
                              os.environ.get("ANTHROPIC_BASE_URL",
                                             "https://openrouter.ai/api/v1"))

    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = JUDGE_PROMPT.format(
        spec=spec,
        gt_script=gt_script,
        agent_script=agent_script,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        raw = resp.choices[0].message.content or ""
        # Extract JSON even if the model wraps it in markdown
        json_m = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_m:
            return json.loads(json_m.group())
        return {"error": f"could not parse JSON from: {raw[:200]}", "score": None}
    except Exception as exc:
        return {"error": str(exc), "score": None}


# ---------------------------------------------------------------------------
# Per-task evaluation
# ---------------------------------------------------------------------------

def evaluate_task(
    task: str,
    agent_task_dir: Path,
    gt_task_dir: Path,
    spec_path: Path,
    run_llm: bool,
    model: str,
) -> dict[str, Any]:
    agent_inputs = agent_task_dir / "inputs"
    gt_inputs    = gt_task_dir    / "inputs"

    # Collect agent .in files
    agent_in_files = sorted(agent_inputs.glob("*.in")) if agent_inputs.exists() else []
    gt_in_files    = sorted(gt_inputs.glob("*.in"))    if gt_inputs.exists()    else []

    result: dict[str, Any] = {
        "task": task,
        "agent_in_files": [str(p.name) for p in agent_in_files],
        "gt_in_files":    [str(p.name) for p in gt_in_files],
        "structural": None,
        "llm_judge": None,
    }

    if not agent_in_files:
        result["structural"] = {"error": "no .in files found in agent output"}
        return result

    # Evaluate the first (or only) .in file
    agent_in = agent_in_files[0]
    gt_in    = gt_in_files[0] if gt_in_files else None

    parsed_agent = parse_in_file(agent_in)
    result["structural"] = structural_check(task, parsed_agent)

    if run_llm:
        spec_text = spec_path.read_text() if spec_path.exists() else "(spec not found)"
        gt_text   = gt_in.read_text() if gt_in else "(no ground truth)"
        agent_text = agent_in.read_text()
        result["llm_judge"] = llm_judge(
            task, spec_text, gt_text, agent_text, model=model,
        )

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate LAMMPS agent outputs against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--agent-run-dir", type=Path, required=True,
                        help="Path to one agent run dir, e.g. data/eval/lammps_plugin/lammps_run1")
    parser.add_argument("--ground-truth-dir", type=Path,
                        default=REPO_ROOT / "data/eval/experiments_lammps_gt",
                        help="GT directory with per-task inputs/ subdirs")
    parser.add_argument("--experiments-dir", type=Path,
                        default=REPO_ROOT / "data/eval/experiments_lammps",
                        help="Task specs directory (instructions.txt per task)")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Where to write per-task JSON results and summary")
    parser.add_argument("--no-llm-judge", action="store_true",
                        help="Skip LLM judge, run structural checks only")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4-6",
                        help="LLM judge model (default: anthropic/claude-sonnet-4-6)")
    parser.add_argument("--tasks", nargs="+",
                        help="Evaluate only these tasks (default: all task subdirs found)")
    args = parser.parse_args()

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Discover tasks
    if args.tasks:
        tasks = args.tasks
    else:
        tasks = sorted(d.name for d in args.agent_run_dir.iterdir() if d.is_dir())

    if not tasks:
        print(f"No task directories found under {args.agent_run_dir}", file=sys.stderr)
        return 1

    all_results: list[dict] = []
    for task in tasks:
        agent_task_dir = args.agent_run_dir / task
        gt_task_dir    = args.ground_truth_dir / task
        spec_path      = args.experiments_dir / task / "instructions.txt"

        if not agent_task_dir.exists():
            print(f"  SKIP  {task}  (not found in agent run dir)")
            continue

        print(f"  eval  {task} ...", end=" ", flush=True)
        result = evaluate_task(
            task=task,
            agent_task_dir=agent_task_dir,
            gt_task_dir=gt_task_dir,
            spec_path=spec_path,
            run_llm=not args.no_llm_judge,
            model=args.model,
        )
        all_results.append(result)

        struct_score = result["structural"].get("score", "?") if result["structural"] else "?"
        llm_score    = result["llm_judge"].get("score", "?") if result["llm_judge"] else "-"
        print(f"structural={struct_score}  llm={llm_score}")

        # Write per-task JSON
        out_path = args.results_dir / f"{task}.json"
        out_path.write_text(json.dumps(result, indent=2))

    # Summary
    summary = {
        "agent_run_dir": str(args.agent_run_dir),
        "tasks_evaluated": len(all_results),
        "per_task": [
            {
                "task": r["task"],
                "structural_score": r["structural"].get("score") if r["structural"] else None,
                "llm_score":        r["llm_judge"].get("score")   if r["llm_judge"]  else None,
            }
            for r in all_results
        ],
    }
    # Average structural score
    struct_scores = [r["structural"]["score"] for r in all_results
                     if r["structural"] and "score" in r["structural"]]
    if struct_scores:
        summary["mean_structural_score"] = round(sum(struct_scores) / len(struct_scores), 3)

    llm_scores = [r["llm_judge"]["score"] for r in all_results
                  if r["llm_judge"] and isinstance(r["llm_judge"].get("score"), (int, float))]
    if llm_scores:
        summary["mean_llm_score"] = round(sum(llm_scores) / len(llm_scores), 2)

    summary_path = args.results_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary written to {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

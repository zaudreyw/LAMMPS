# LAMMPS Agent Harness: Design, Implementation, and Evaluation

**Author:** Audrey  
**Date:** May 2026  
**Status:** Run 3 complete

---

## Overview

This document describes the design and implementation of an AI agent evaluation harness for LAMMPS (Large-scale Atomic/Massively Parallel Simulator). The work adapts an existing agent-evaluation infrastructure — originally built for GEOS, a reservoir simulation code — to evaluate how well large language model (LLM) agents can author physically correct LAMMPS molecular dynamics input scripts from natural-language specifications.

The central research question is: **what scaffolding helps an LLM agent write better LAMMPS input scripts?** The harness implements a controlled ablation study across three agent configurations that isolate the contribution of (1) retrieval-augmented generation (RAG) over LAMMPS documentation, and (2) structural self-verification hooks that give the agent real-time feedback on its output.

---

## Background

### What is LAMMPS?

LAMMPS is an open-source molecular dynamics code maintained by Sandia National Laboratories. It simulates collections of atoms or molecules by integrating Newton's laws of motion forward in time. Given an initial geometry and a set of interatomic forces (a "force field"), LAMMPS steps the system forward — typically in femtosecond increments — and records positions, velocities, energies, and stresses.

LAMMPS is domain-general: the same code handles metals, polymers, proteins, granular materials, and coarse-grained models. What changes between simulations is the force field and the thermodynamic ensemble (what quantities are held constant during the run). Users author plain-text input scripts (`.in` files) that specify all of this in a command-by-command format where **order matters**.

### Why this is a hard task for LLMs

LAMMPS input scripts have several properties that make them difficult for LLMs to author correctly:

- **Strict command ordering.** Initialization commands (`units`, `atom_style`) must precede geometry setup, which must precede pair potentials, which must precede fixes, which must precede the `run` command. Violations are syntax errors.
- **Large vocabulary.** LAMMPS has ~300 commands, each with its own keyword arguments. Many keywords look plausible but do not exist — the agent must verify against documentation rather than interpolate.
- **Precise numerical parameters.** Simulation correctness depends on exact values (cutoffs, timestep, thermostat damping constants, atomic charges) that vary by unit system.
- **Geometry consistency.** Box dimensions, atom types, and coordinate system centering must be internally consistent — errors here produce simulations that silently run with the wrong geometry.
- **No equivalent of a schema validator.** Unlike XML-based simulators, there is no lightweight dry-run check short of executing the LAMMPS binary itself.

---

## System Architecture

The harness runs each agent inside a Docker container. Inside the container, the agent has access to:

- `/workspace/inputs/` — where it writes output `.in` scripts and any auxiliary files (data files, potential files)
- `/lammps_lib/` — a read-only bind mount of the LAMMPS source tree (examples and documentation)
- MCP servers (if enabled) — providing RAG search and validation tools

The host orchestrator manages task dispatch, environment variable injection, MCP server pre-flight checks, and result collection.

```
Host (orchestrator)
│
├── Task instructions → injected into agent system prompt
├── AGENTS_lammps.md → agent system prompt
├── plugin_lammps/ → plugin manifest, hooks, MCP servers
│
└── Docker container
    ├── Agent (Claude Code, claude_native runner)
    │   ├── MCP: lammps-rag (search_navigator, search_technical, search_commands)
    │   ├── MCP: lammps-validate (validate_lammps_input) [optional]
    │   ├── Hook: PostToolUse → verify_lammps_post_write.py
    │   └── Hook: Stop → verify_outputs.py
    ├── /workspace/inputs/    ← agent writes here
    ├── /workspace/outputs/
    └── /lammps_lib/          ← read-only LAMMPS source
```

---

## What Was Built

### 1. LAMMPS Plugin (`plugin_lammps/`)

A self-contained plugin directory that packages all LAMMPS-specific components. The directory structure is:

```
plugin_lammps/
├── .claude-plugin/
│   └── plugin.json                    ← plugin manifest + MCP server declarations
├── scripts/
│   ├── build_lammps_vector_db.py      ← offline indexing script (builds ChromaDB)
│   ├── lammps_rag_mcp.py              ← RAG MCP server (3 search tools)
│   └── lammps_validate_mcp.py         ← validation MCP server (1 tool)
├── hooks/
│   ├── hooks.json                     ← hook configuration (Stop + PostToolUse)
│   ├── verify_outputs.py              ← Stop hook: end-of-turn structural validation
│   └── verify_lammps_post_write.py    ← PostToolUse hook: immediate write validation
├── skills/
│   └── lammps-rag/SKILL.md            ← skill definition for RAG tool invocation
├── LAMMPS_PRIMER_minimal.md           ← agent quick-reference primer
└── README.md
```

#### 1a. RAG Vector Database Builder (`scripts/build_lammps_vector_db.py`)

An offline indexing script that processes the LAMMPS source tree and populates a **ChromaDB** persistent vector database. The database has three collections, each targeting a different information need:

| Collection | Source material | Use case |
|---|---|---|
| `lammps_navigator` | Conceptual/howto RST pages from `doc/src/` | "How do I run NVT?" / "Which fix style for rigid bodies?" |
| `lammps_technical` | Example `.in` scripts from `examples/` | "Show me a Lennard-Jones melt example" |
| `lammps_commands` | Command reference RST pages from `doc/src/` | "What are the arguments to `fix nvt`?" / "Exact `pair_style lj/cut` syntax" |

**Indexing approach:**

- *Navigator and commands*: RST pages are chunked by section heading (H1/H2), keeping 300–600 tokens per chunk. Metadata stored per chunk includes `source_path`, `title`, `breadcrumbs`, and `chunk_type` (with syntax blocks tagged `chunk_type=syntax`).
- *Technical (example scripts)*: Raw LAMMPS commands are semantically sparse and embed poorly as-is. To improve retrieval quality, each example script is processed into a "shadow description" — a 2–4 sentence natural-language summary of what the simulation does, which physics model it uses, and what output it produces. The shadow text is embedded; the actual script path is stored as metadata and retrieved on match.
- *Embedding model*: `qwen/qwen3-embedding-8b` via OpenRouter. The `--shadow-mode llm` flag optionally uses an LLM to generate higher-quality shadow descriptions at the cost of API calls.

**Ground-truth leakage prevention**: The RAG server reads the `EXCLUDED_GT_IN_FILENAMES` environment variable (a comma-separated list of `.in` filenames) and filters those files out of all search results. The orchestrator populates this variable per task from the ground-truth directory, ensuring the agent cannot retrieve the reference answer through the search interface.

#### 1b. RAG MCP Server (`scripts/lammps_rag_mcp.py`)

An MCP server built with `fastmcp` that exposes three search tools over the ChromaDB collections:

- `mcp__lammps-rag__search_navigator(query, n_results)` — searches the conceptual documentation collection
- `mcp__lammps-rag__search_technical(query, n_results)` — searches the example scripts collection
- `mcp__lammps-rag__search_commands(query, n_results)` — searches the command reference collection

The server is launched via `uv run --script` with no installation step required. The agent is instructed in its system prompt to call at least one RAG tool before writing any `.in` script, and to prefer reading the full matched example from `/lammps_lib/examples/` rather than trusting the snippet alone.

#### 1c. Validation MCP Server (`scripts/lammps_validate_mcp.py`)

An MCP server exposing a single tool: `validate_lammps_input(in_path)`. This provides mid-task validation that the agent can call proactively, before ending its turn, to catch errors without waiting for the Stop hook.

Validation runs two tiers in sequence:

1. **Structural check** (always): verifies that `units`, `atom_style`, and at least one `run`/`minimize` command are present in the file.
2. **Binary check** (when LAMMPS is in PATH): invokes `lammps -skiprun -nocite -log none -screen none -in <file>`. The `-skiprun` flag (available since LAMMPS 15Dec2020) parses and validates the script without executing the simulation. Missing data files or potential files are reported as "missing file" errors, distinguished from syntax errors, so the agent knows what to fix. Falls back to Tier 1 only if `-skiprun` is not recognized.

#### 1d. Stop Hook (`hooks/verify_outputs.py`)

Fires when the agent ends its turn. Checks `/workspace/inputs/` for structurally valid `.in` files and blocks with targeted feedback if validation fails. Three tiers:

1. **Presence**: at least one `.in` file must exist
2. **Structure**: each file must contain `units`, `atom_style`, and at least one of `run`/`minimize`/`rerun`/`tad`/`neb`/`prd` (and others)
3. **Binary** (opt-in via `LAMMPS_HOOK_LAMMPS_CHECK=1`): runs the LAMMPS binary against each file

When a file fails, the hook returns `decision: "block"` with a specific message listing the missing commands. The agent re-enters and receives this message as feedback. A configurable retry budget (default: 2) prevents infinite loops. An optional `LAMMPS_HOOK_SELF_REFLECT=1` mode prompts the agent to review its own script one additional time after all checks pass.

#### 1e. PostToolUse Hook (`hooks/verify_lammps_post_write.py`)

Fires immediately after every `Write`, `Edit`, or `MultiEdit` tool call on a `.in` file inside `/workspace/inputs/`. Runs the same structural checks as the Stop hook but with a 15-second timeout and no retry budget — every problematic write fires until fixed.

This gives the agent feedback within seconds of a bad write, rather than only at end-of-turn. The logic is shared by dynamically importing from `verify_outputs.py` at runtime to avoid code duplication.

---

### 2. Agent System Prompt (`run/AGENTS_lammps.md`)

A dedicated system prompt for LAMMPS tasks, distinct from the GEOS system prompt. It establishes:

- **Role**: LAMMPS expert authoring input scripts from natural-language specs. Does not attempt to run LAMMPS.
- **Workspace layout**: writes all files to `/workspace/inputs/`; LAMMPS source is at `/lammps_lib/` (read-only).
- **Inline primer**: command ordering rules, physics pattern table (NVE/NVT/NPT/GCMC/rigid/charged), units reference table, and safety rules.

Key safety rules baked into the prompt:
- `units` and `atom_style` must be the very first commands
- `pair_coeff` atom-type indices are 1-based and must exactly match the count in `create_box` or the data file
- All dump paths should be absolute (e.g., `/workspace/inputs/dump.lammpstrj`)
- Never invent keyword names — verify against `/lammps_lib/doc/src/`

A standalone minimal primer (`plugin_lammps/LAMMPS_PRIMER_minimal.md`) is also available for use with `--strip-baked-primer`, covering the same content in a format that can be hot-swapped for ablation experiments on primer content.

---

### 3. Runner Infrastructure Extensions (`src/runner/`)

The existing runner was GEOS-specific throughout. The following files were extended to support LAMMPS agents via a `lammps_mode` flag:

#### `src/runner/agents.py` — Agent definitions

Six LAMMPS agent configurations were added, all tagged `lammps_mode: True`. The `lammps_mode` flag routes them through the LAMMPS code path in the orchestrator instead of the GEOS code path.

| Agent key | RAG | Hook | Validate MCP | Purpose |
|---|---|---|---|---|
| `lammps_vanilla` | — | — | — | Baseline: model knowledge only |
| `lammps_plugin` | ✓ | ✓ | — | Full stack |
| `lammps_plugin_no_rag` | — | ✓ | — | Hook contribution ablation |
| `lammps_plugin_no_hook` | ✓ | — | — | RAG contribution ablation |
| `lammps_plugin_validate` | ✓ | ✓ | ✓ | Full stack + validate tool |
| `lammps_plugin_validate_no_rag` | — | ✓ | ✓ | Validate tool without RAG |

#### `src/runner/cli.py` — Command-line interface

Two new flags added:

- `--agents-md-path PATH` — overrides the default `run/AGENTS.md` system prompt file. Pass `run/AGENTS_lammps.md` to use the LAMMPS harness prompt. This decouples the system prompt from the agent definition, enabling prompt ablation without changing agent config.
- `--lammps-lib-dir DIR` — path to the LAMMPS source tree on the host. Mounted read-only at `/lammps_lib` in the Docker container. Validation is skipped for GEOS-specific checks when only LAMMPS agents are selected (and vice versa).

#### `src/runner/orchestrator.py` — Task orchestration

The `run_task()` function now detects `lammps_mode` at the start and branches accordingly:

**GEOS path** (unchanged): creates a filtered copy of the GEOS source tree with ground-truth XML files and RST tutorial pages excluded via hardlink filtering, then mounts it at `/geos_lib`.

**LAMMPS path** (new): skips the filtered-copy step entirely (the full LAMMPS source is mounted read-only at `/lammps_lib`). Ground-truth leakage is prevented instead by reading the per-task GT directory, collecting `.in` filenames, and passing them to the RAG server via `EXCLUDED_GT_IN_FILENAMES` so they are filtered from search results.

#### `src/runner/docker_cmd.py` — Docker command construction

New functions for LAMMPS agents:

- `build_lammps_native_command()` — builds the `docker run` command with `/lammps_lib` mounted read-only instead of `/geos_lib`
- `build_lammps_native_env()` — constructs the environment variable set (API keys, `LAMMPS_VECTOR_DB_DIR`, `EXCLUDED_GT_IN_FILENAMES`, hook enable flags)
- `preflight_lammps_mcp()` — verifies that the RAG MCP server starts without error before the agent run begins

#### `src/runner/claude_settings.py` — Agent settings generation

New functions:

- `write_lammps_claude_settings()` — writes the per-run Claude Code settings file enabling the correct hooks and tools for the agent configuration
- `write_lammps_mcp_config()` — writes the per-run MCP config pointing to the RAG and validate servers

#### `src/runner/constants.py` — Shared constants

Added: `DEFAULT_LAMMPS_LIB_DIR`, `DEFAULT_LAMMPS_VECTOR_DB_DIR`, `LAMMPS_EXPERIMENTS_DIR`.

#### `src/runner/prompts/` — Prompt fragments

Added:
- `native_plugin_prefix_lammps.txt` — injected at the start of the system prompt when the plugin is enabled, instructing the agent to use RAG tools by their MCP names
- `rag_instructions_lammps.txt` — explains when to use each of the three RAG tools (navigator for concepts, technical for examples, commands for exact syntax)

---

### 4. Evaluation Tasks (`data/eval/experiments_lammps/`)

Four tasks were defined, covering a range of physics complexity:

| Task | Physics | Complexity |
|---|---|---|
| `lj_melt_minimal` | Lennard-Jones argon melt, 256 atoms, NVT, 2000 steps | Minimal spec — tests basic competence |
| `lj_melt` | LJ argon melt with detailed spec: FCC lattice, NVE pre-equilibration, NVT production, dump output | Moderate — tests multi-phase setup |
| `nvt_water` | SPC/E water, 216 molecules, PPPM electrostatics, SHAKE constraints, NVT at 300 K | High — tests bonded topology, charges, long-range electrostatics |
| `crack_2d` | 2D crack propagation in LJ solid, pre-notch geometry, boundary loading via `fix move`, stress/atom output | Highest — tests 2D simulation, geometry centering, stress computation |

Each task directory contains `instructions.txt` (the natural-language spec given to the agent) and an `inputs/` subdirectory with any auxiliary files needed (e.g., potential files). Ground-truth reference scripts are stored separately in `data/eval/experiments_lammps_gt/` and are never exposed to the agent.

---

### 5. Evaluation Script (`scripts/eval/batch_lammps_evaluate.py`)

Scores each agent run on two metrics:

**Structural score (0.0–1.0):** Regex and keyword checks on the generated `.in` file. Checks include: presence of `units`, `atom_style`, `dimension` (for 2D tasks), correct `run` step count, correct unit system, presence of an integrator fix (`nve`/`nvt`/`npt`). Score is the fraction of checks passed.

**LLM judge score (1–10):** A separate LLM call that receives the agent's script, the task specification, and instructions to score on physical correctness, syntactic correctness, and completeness. Returns a numeric score with a rationale paragraph and a list of key issues. Results are written as JSON per task and aggregated into a `summary.json`.

---

## Experimental Results — Run 3

Three agents were evaluated across all four tasks.

### Summary

| Agent | Mean Structural | Mean LLM |
|---|---|---|
| `lammps_vanilla` | 0.969 | ~5.25† |
| `lammps_plugin_no_rag` | **1.000** | 5.25 |
| `lammps_plugin` | 0.969 | **6.75** |

†The vanilla mean LLM score is reported as 7.0 in `summary.json` but is computed over only 3 tasks — the LLM judge for `crack_2d` returned a markdown-formatted response instead of JSON, causing a parse failure and a null score. The corrected comparable mean is approximately 5.25.

### Per-Task Results

| Task | Agent | Structural | LLM |
|---|---|---|---|
| `crack_2d` | vanilla | 0.875 | null (judge parse error) |
| `crack_2d` | plugin_no_rag | **1.000** | 2 |
| `crack_2d` | plugin | 0.875 | 3 |
| `lj_melt` | vanilla | 1.000 | 9 |
| `lj_melt` | plugin_no_rag | 1.000 | 7 |
| `lj_melt` | plugin | 1.000 | **9** |
| `lj_melt_minimal` | vanilla | 1.000 | 6 |
| `lj_melt_minimal` | plugin_no_rag | 1.000 | 6 |
| `lj_melt_minimal` | plugin | 1.000 | **9** |
| `nvt_water` | vanilla | 1.000 | 6 |
| `nvt_water` | plugin_no_rag | 1.000 | 6 |
| `nvt_water` | plugin | 1.000 | 6 |

### Key Findings

**1. Structural scores are a weak quality signal.** `lammps_plugin_no_rag` achieves a perfect structural score on `crack_2d` yet earns an LLM score of 2. The LLM judge found critical LAMMPS syntax errors — a non-existent `strip` region style, malformed `create_box` syntax, and three atom types declared but only one used — that regex checks cannot detect. This confirms that structural scoring needs richer semantic checks to be meaningful.

**2. RAG retrieval demonstrably improves physics-level quality.** The `lammps_plugin` (RAG + hooks) outperforms `lammps_plugin_no_rag` (hooks only) by +3 LLM points on `lj_melt_minimal` and +2 on `lj_melt`. The RAG-augmented agent finds and adapts real reference scripts from the LAMMPS examples library; without RAG, the agent relies on parametric knowledge and produces scripts with non-standard geometry constructions or redundant calculations that the judge penalizes.

**3. `crack_2d` is the universal failure mode.** It is the only task where structural scores drop below 1.0, and it consistently produces the lowest LLM scores across all agents. Two independent failure patterns emerge:
   - *Missing integrator*: Vanilla and plugin both omit `fix nve all nve` despite the task specifying "NVE dynamics." The structural checker correctly flags this, but both agents fail to correct it.
   - *Coordinate geometry error*: The task specifies a box centered at the origin (x: −50 to 50, y: −20 to 20) with a notch from x = −50 to x = 0 and boundary strips at y > 19 and y < −19. Agents consistently generate 0-anchored boxes (x: 0 to 100, y: 0 to 40), which makes the notch and boundary-strip region coordinates land outside or at the edge of the box — resulting in a simulation that runs but does not model the intended crack geometry.

**4. `nvt_water` plateaus at LLM=6 for all agents.** The shared ceiling likely reflects a consistent omission: the task specifies that output files go to `/workspace/inputs/`, but agents tend to use relative paths or omit explicit output paths in `dump` and `write_data` commands.

---

## Identified Gaps and Improvement Plan

### Priority 1 — Fix the LLM judge JSON parsing bug

The judge for vanilla/`crack_2d` returned a markdown-formatted response, not JSON, causing a null score. Without this fix, cross-agent mean LLM comparisons are not valid. **Fix:** add a fallback that strips markdown fences before JSON parsing, or use a structured-output API call.

### Priority 2 — Resolve `crack_2d` coordinate system failures

The most impactful quality gap. **Two sub-fixes:**
- Add explicit instruction to the task prompt and/or the LAMMPS primer: "the simulation box must be centered at the origin; use `region box block -50 50 -20 20 -0.5 0.5`."
- Add a RAG document or few-shot example covering 2D crack simulations with a centered box. The closest existing LAMMPS reference (`examples/crack/in.crack`) uses the correct centered geometry and is already in the vector database — the agent should be finding it.

### Priority 3 — Add a `crack_2d` ground-truth input file

`data/eval/experiments_lammps_gt/crack_2d/inputs/` currently exists but is empty. Adding a canonical `in.crack_2d` script would: (a) enable file-diff based scoring as an additional metric, (b) make the RAG technical collection more useful (it can index the GT once evaluation is over), and (c) give the LLM judge a concrete reference to compare against.

### Priority 4 — Strengthen `crack_2d` structural checks

The 8-check suite passes a script with critical errors. Add checks for:
- Integrator type consistency (NVE task → requires `fix nve`, not just any integrator fix)
- Box coordinate centering (check that `xlo` is negative for centered-box tasks)
- Atom type count consistency (`create_box N` must equal the number of distinct `pair_coeff` lines)
- `compute stress/atom` presence when the task requests per-atom stress output

### Priority 5 — Fix `nvt_water` output path

All agents score 6 on `nvt_water`. The LLM judge for `lj_melt_minimal` flagged missing output paths as a recurring issue. Add to the LAMMPS primer and task instructions: dump paths must be absolute (e.g., `/workspace/inputs/dump.water.lammpstrj`). This is a one-line fix with a likely +1 to +2 LLM point gain.

### Priority 6 — Aggregate results across multiple runs

Run 3 provides a single sample per agent per task. Given the observed variance (e.g., `crack_2d` LLM scores: 2, 3 across agents on a 10-point scale), single-run conclusions are statistically fragile. Run 1 and Run 2 data already exist in `data/eval/`. Aggregating across all three runs and reporting mean ± standard deviation would substantially strengthen any claims about relative agent performance.

---

## Repository Layout (LAMMPS-relevant files)

```
repo3/
├── plugin_lammps/               ← LAMMPS plugin (RAG, hooks, validation)
│   ├── .claude-plugin/plugin.json
│   ├── scripts/
│   │   ├── build_lammps_vector_db.py
│   │   ├── lammps_rag_mcp.py
│   │   └── lammps_validate_mcp.py
│   ├── hooks/
│   │   ├── hooks.json
│   │   ├── verify_outputs.py
│   │   └── verify_lammps_post_write.py
│   └── LAMMPS_PRIMER_minimal.md
├── run/
│   └── AGENTS_lammps.md         ← LAMMPS agent system prompt
├── src/runner/
│   ├── agents.py                ← LAMMPS agent definitions (lammps_vanilla, etc.)
│   ├── cli.py                   ← --agents-md-path, --lammps-lib-dir flags
│   ├── orchestrator.py          ← lammps_mode routing
│   ├── docker_cmd.py            ← LAMMPS Docker command builders
│   ├── claude_settings.py       ← LAMMPS settings/MCP config writers
│   ├── constants.py             ← LAMMPS path constants
│   └── prompts/
│       ├── native_plugin_prefix_lammps.txt
│       └── rag_instructions_lammps.txt
├── data/eval/
│   ├── experiments_lammps/      ← task instructions (lj_melt, nvt_water, crack_2d, …)
│   ├── experiments_lammps_gt/   ← ground-truth reference scripts
│   ├── lammps_vanilla/          ← agent run outputs
│   ├── lammps_plugin/
│   ├── lammps_plugin_no_rag/
│   └── lammps_scores/           ← evaluation results (per-task JSON + summary)
├── scripts/eval/
│   └── batch_lammps_evaluate.py ← evaluation script
└── writing/
    ├── LAMMPS_OVERVIEW.md       ← background reading: what LAMMPS is, task descriptions
    └── LAMMPS_Agent_Harness.md  ← this document
```

---

## How to Reproduce

### Build the vector database (one-time)

```bash
cd plugin_lammps
export OPENROUTER_API_KEY=<your-key>
export LAMMPS_VECTOR_DB_DIR=/data/shared/lammps_agent_data/data/vector_db

uv run scripts/build_lammps_vector_db.py \
    --lammps-src /path/to/lammps \
    --vector-db-dir $LAMMPS_VECTOR_DB_DIR
```

### Run an agent evaluation

```bash
for AGENT in lammps_vanilla lammps_plugin_no_rag lammps_plugin; do
    uv run python -m src.runner.cli \
        --run run3 \
        --agents $AGENT \
        --agents-md-path run/AGENTS_lammps.md \
        --plugin-dir plugin_lammps \
        --experiments-dir data/eval/experiments_lammps \
        --ground-truth-dir data/eval/experiments_lammps_gt \
        --lammps-lib-dir /path/to/lammps \
        --vector-db-dir /data/shared/lammps_agent_data/data/vector_db
done
```

### Score the results

```bash
for AGENT in lammps_vanilla lammps_plugin_no_rag lammps_plugin; do
    uv run python scripts/eval/batch_lammps_evaluate.py \
        --agent-run-dir data/eval/${AGENT}/run3 \
        --ground-truth-dir data/eval/experiments_lammps_gt \
        --experiments-dir data/eval/experiments_lammps \
        --results-dir data/eval/lammps_scores/${AGENT}/run3
done
```

Results are written to `data/eval/lammps_scores/<agent>/run3/` as one JSON file per task and a `summary.json` with aggregated means.

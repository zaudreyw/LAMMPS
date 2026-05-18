# LAMMPS Simulation Fixes — 2026-05-18

## 1. Mine Full (Input, Output) Example List

### What was missing
There was no machine-readable manifest linking eval tasks to their source files. The 4 existing tasks were hand-authored with no record of which LAMMPS example each was derived from.

### What I did
Created `data/eval/lammps_example_pairs.jsonl` — an 8-line manifest mapping every eval task to its corresponding source `.in` file(s) in the LAMMPS examples directory:

```
{"task_id": "lj_melt",          "lammps_example_relpaths": ["examples/melt/in.melt"]}
{"task_id": "crack_2d",         "lammps_example_relpaths": ["examples/crack/in.crack"]}
{"task_id": "nvt_water",        "lammps_example_relpaths": ["examples/water/in.water", "examples/rhodo/in.rhodo"]}
{"task_id": "lj_solid",         "lammps_example_relpaths": ["examples/melt/in.melt"]}
{"task_id": "couette_flow",     "lammps_example_relpaths": ["examples/flow/in.flow.couette"]}
{"task_id": "lj_indent",        "lammps_example_relpaths": ["examples/indent/in.indent"]}
{"task_id": "msd_diffusion",    "lammps_example_relpaths": ["examples/diffuse/in.diffuse"]}
{"task_id": "uniaxial_tension", "lammps_example_relpaths": ["examples/deformation/in.deformation"]}
```

The **input** for each task is its `instructions.txt` (the natural-language specification the agent receives). The **output** is the GT `.in` script in `experiments_lammps_gt/<task>/inputs/`.

---

## 2. Report Mining Method (Differences from GEOS)

### What was missing
No mining script existed. All 4 original tasks were hand-authored with no automated pipeline and no documentation of the method gap vs. GEOS.

### What I did
Created `plugin_lammps/scripts/mine_lammps_examples.py` — a standalone miner that scans any LAMMPS source tree and produces a full `lammps_example_pairs.jsonl`. Running it also prints a method summary comparing LAMMPS and GEOS mining.

### Key differences from GEOS mining

| Dimension | GEOS | LAMMPS |
|---|---|---|
| Source of task spec | RST tutorial prose pages | `.in` script content + README |
| Parsing target | RST headings, parameter tables, prose | `units`, `atom_style`, `pair_style`, `fix` commands |
| Task ID mapping | `task_id` → RST file path | `task_id` → source `.in` file path |
| Blocking unit | RST file + XML basename + variant siblings | Source `.in` file relative path |
| Variant expansion | 9 XML suffix patterns (`_base`, `_smoke`, etc.) | N/A — LAMMPS `.in` files have no variant naming convention |
| Automation | Semi-automated (RST miner existed in GEOS repo) | Newly written for this project |

The fundamental difference: GEOS tasks come from structured tutorial documentation, so the spec is extracted from prose. LAMMPS examples have minimal READMEs and no structured tutorial format, so the task spec is auto-generated from parsing the `.in` script metadata itself.

---

## 3. Report Counts

| Item | Count |
|---|---|
| Total eval tasks | 9 |
| Tasks with ground truth | 8 |
| Tasks without ground truth | 1 (`lj_melt_minimal`) |
| GT `.in` files | 8 |
| Tasks in `lammps_example_pairs.jsonl` | 8 |
| LAMMPS agents defined | 9 (6 Claude, 3 DeepSeek) |

Counts are now derivable from the filesystem directly:
```bash
find data/eval/experiments_lammps -name "instructions.txt" | wc -l    # 9 tasks
find data/eval/experiments_lammps_gt -name "in.*" | wc -l             # 8 GT files
wc -l data/eval/lammps_example_pairs.jsonl                             # 8 mapped
```

---

## 4. Double Check RAG vs. Agentic File Navigation

### What was wrong
The two retrieval paths were not symmetrically decontaminated:

- **RAG path:** The RAG MCP server (`lammps_rag_mcp.py`) already excluded GT filenames from search results via the `EXCLUDED_GT_IN_FILENAMES` environment variable. ✓
- **File navigation path:** The full unfiltered LAMMPS source was mounted at `/lammps_lib` inside the container. An agent using `Bash` or `Read` could open `examples/melt/in.melt` directly and copy it verbatim. ✗

This meant the `lammps_plugin_no_rag` ablation condition (RAG off, hook on) was not actually testing "can the agent solve the task without RAG" — it was testing "can the agent find and copy the exact answer file."

### What I did
The filesystem-level filtering implemented in Section 5 closes this gap. Both paths are now decontaminated identically, so the RAG vs. file-navigation comparison is valid. The relevant comment added to `orchestrator.py`:

```python
# LAMMPS decontamination:
#   1. RAG-level: GT .in basenames passed via EXCLUDED_GT_IN_FILENAMES so
#      the RAG server filters them from search results.
#   2. Filesystem-level: hardlink copy with source .in files removed,
#      so the agent cannot read them via Bash/Read either.
#   Both layers are needed to decontaminate RAG vs. agentic file navigation.
```

---

## 5. Implement Containerization Decontamination Process

### What was missing
The GEOS runner had full decontamination: a hardlink copy of the GEOS source with GT XML files (and variant siblings and the source RST tutorial) removed before Docker mount. The LAMMPS runner had none — it mounted the full LAMMPS source read-only with no filtering at the filesystem level.

Old code in `orchestrator.py`:
```python
# No file-system filtering is done — the whole LAMMPS lib is mounted read-only.
filtered_geos = lammps_root  # placeholder, never filtered
cleanup_filtered_copy = False
```

### What I did

**`src/runner/contamination.py`** — Added four new functions:

- `_load_lammps_example_pairs()` — reads `lammps_example_pairs.jsonl` into a `task_id → [relpaths]` dict
- `get_blocked_in_files_for_task(task_id, gt_dir)` — returns both the GT basename list (for RAG) and the source relpaths (for filesystem filtering)
- `create_filtered_lammps_copy(lammps_src, blocked_in_source_relpaths, tmp_parent)` — hardlinks the entire LAMMPS source tree into a temp directory, skipping the listed `.in` files
- `cleanup_filtered_lammps_copy(lammps_copy)` — removes the temp tree after the run completes

**`src/runner/constants.py`** — Added two new constants:
- `TEMP_LAMMPS_PARENT` — filesystem location for temp filtered copies (should be on same filesystem as the LAMMPS source for efficient hardlinks)
- `DEFAULT_LAMMPS_EXAMPLE_PAIRS` — default path to `lammps_example_pairs.jsonl`

**`src/runner/orchestrator.py`** — Replaced the no-op block with real filtering:

```python
# NEW: call get_blocked_in_files_for_task to get both lists
lammps_blocked = get_blocked_in_files_for_task(task_name, ground_truth_dir)
blocked_in_filenames = lammps_blocked["blocked_in_basenames"]         # → RAG env var
blocked_source_relpaths = lammps_blocked["blocked_in_source_relpaths"]  # → FS filter

# Create filtered hardlink copy if there are source files to block
if not dry_run and blocked_source_relpaths:
    filtered_lammps = create_filtered_lammps_copy(
        lammps_root,
        blocked_in_source_relpaths=blocked_source_relpaths,
        tmp_parent=TEMP_LAMMPS_PARENT,
    )
    cleanup_filtered_copy = True

# Mount the filtered copy (or full source if nothing to block)
lammps_mount_dir = filtered_lammps or lammps_root
```

The filtered path is then passed to `build_lammps_native_command(lammps_lib_dir=lammps_mount_dir, ...)` so Docker mounts only the sanitized tree at `/lammps_lib`. The `finally` block calls `cleanup_filtered_lammps_copy` to remove the temp copy after the run.

### Ensure exact tutorial doc is not available during corresponding experiment

This is what `lammps_example_pairs.jsonl` drives at runtime. Example: when `lj_melt` runs, `examples/melt/in.melt` is removed from the mounted `/lammps_lib`. The agent can still navigate all other examples (crack, water, indent, etc.) — only the specific source file the task was derived from is absent. This matches the GEOS approach of blocking only the task-relevant RST page, not the entire documentation.

---

## 6. Scale Experiments (More Simulations)

### What was missing
Only 4 tasks existed (3 with GT). The eval matrix was too small to draw statistically meaningful conclusions.

### What I did
Added 5 new tasks, bringing the total to 9 (8 with GT). Each has both a precisely-parameterized `instructions.txt` and a complete GT `.in` script. All are fully self-contained — no auxiliary data files required.

| Task | Physics | Key LAMMPS Features Tested |
|---|---|---|
| `lj_solid` | LJ FCC crystal equilibration at T=0.1 | `fix nvt`, low-temperature solid |
| `couette_flow` | NEMD Couette flow | `fix nvt/sllod`, `fix deform xy erate`, `compute temp/deform` |
| `lj_indent` | 2D nanoindentation | `dimension 2`, `fix indent sphere`, equal-style variable |
| `msd_diffusion` | MSD / diffusion coefficient | `compute msd com yes`, NVE production run |
| `uniaxial_tension` | Tensile deformation of FCC crystal | `fix deform z erate`, lateral `fix npt` relaxation |

The 5 tasks were chosen to cover distinct physics regimes (equilibrium solid, NEMD flow, surface mechanics, transport, deformation) and distinct LAMMPS subsystems, so that agent performance variation across tasks reflects physics difficulty rather than task similarity.

---

## 7. Try Different Agents Including DeepSeek v4 Flash

### What was missing
All LAMMPS agents used the same base model (`minimax/minimax-m2.7` via OpenRouter). There was no cross-model comparison for LAMMPS.

### What I did
Added three new agent entries to `src/runner/agents.py` in a dedicated "LAMMPS cross-model" section:

| Agent key | Plugin | RAG | Hook | Purpose |
|---|---|---|---|---|
| `lammps_deepseek_vanilla` | off | off | off | Baseline: raw DeepSeek performance |
| `lammps_deepseek_plugin` | on | on | on | Full stack with DeepSeek |
| `lammps_deepseek_no_rag` | on | off | on | Hook only — tests file navigation vs RAG |

Model: `deepseek/deepseek-chat-v3-0324` (DeepSeek v4 flash, accessed via OpenRouter at `ANTHROPIC_BASE_URL=https://openrouter.ai/api`). Uses the same `ANTHROPIC_AUTH_TOKEN` credential as all other agents — no new secrets needed.

The three-cell design mirrors the existing Claude ablation (`lammps_vanilla`, `lammps_plugin`, `lammps_plugin_no_rag`), so DeepSeek and Claude results are directly comparable across all scaffold conditions.

To run the DeepSeek agents, use the same CLI flags as for Claude LAMMPS agents:
```bash
python -m src.runner.cli \
  --run lammps_deepseek_exp1 \
  --agents lammps_deepseek_vanilla lammps_deepseek_plugin lammps_deepseek_no_rag \
  --agents-md-path run/AGENTS_lammps.md \
  --plugin-dir plugin_lammps \
  --vector-db-dir /data/shared/lammps_agent_data/data/vector_db \
  --lammps-lib-dir /path/to/lammps \
  --experiments-dir data/eval/experiments_lammps \
  --ground-truth-dir data/eval/experiments_lammps_gt
```

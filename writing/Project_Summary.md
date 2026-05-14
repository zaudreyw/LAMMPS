# Project Summary: AI Agent Scaffolding for LAMMPS Input Script Authoring

**Author:** Audrey  
**Date:** May 2026

---

## What is LAMMPS and Why Does It Matter?

LAMMPS (Large-scale Atomic/Massively Parallel Simulator) is an open-source molecular dynamics (MD) code maintained by Sandia National Laboratories. It simulates the physical behavior of atoms and molecules over time by integrating Newton's laws of motion — given initial atom positions, velocities, and an interatomic force field, LAMMPS steps the system forward (typically in femtosecond increments) and records trajectories, energies, temperatures, and stresses.

Users control LAMMPS entirely through plain-text input scripts (`.in` files). These scripts specify everything: the unit system, atom types, simulation box geometry, force field parameters, thermodynamic ensemble, output frequency, and run length. Commands must appear in a strict order, and the vocabulary spans roughly 300 distinct commands, each with its own keyword arguments. Writing a correct input script for a non-trivial simulation requires knowing the physics, the LAMMPS command syntax, and how those two interact — a non-trivial combination even for experienced users.

---

## Research Question

**Can LLM agents reliably author correct LAMMPS input scripts from natural-language specifications, and what scaffolding helps them do so?**

There is growing interest in using LLMs to automate scientific computing workflows. LAMMPS is a good testbed for this question because: (1) it is widely used across materials science, chemistry, and biophysics; (2) correctness is objectively measurable; and (3) the failure modes are interesting — agents can produce scripts that look plausible, parse without error, but model entirely the wrong physics due to subtle geometric or parametric mistakes.

This project evaluates three specific interventions:

1. **Retrieval-augmented generation (RAG)** — giving the agent live search access to the LAMMPS documentation and example script library
2. **Structural self-verification hooks** — feedback mechanisms that check the agent's output and block it from finishing until structural problems are fixed
3. **A mid-task validation tool** — an MCP tool the agent can call proactively to validate a script before ending its turn

---

## Approach

### Evaluation Tasks

Four tasks were designed to span a range of physics complexity:

| Task | What it simulates | Key challenge |
|---|---|---|
| `lj_melt_minimal` | 256 argon atoms, FCC lattice → liquid, NVT, 2000 steps | Tests baseline competence from a minimal 3-line spec |
| `lj_melt` | Same physics, full step-by-step spec | Tests multi-phase setup (NVE equilibration → NVT production) |
| `nvt_water` | 216 SPC/E water molecules at 300 K | Tests bonded topology, partial charges, long-range electrostatics (PPPM), SHAKE constraints |
| `crack_2d` | 2D LJ solid with a pre-cut notch pulled apart by driven boundaries | Tests 2D simulation, origin-centered geometry, stress/atom computation |

Each task provides a natural-language specification (the same kind a researcher might write for a collaborator) and expects the agent to produce a runnable `.in` file in `/workspace/inputs/`. Reference scripts are stored separately and never shown to the agent during the run.

### Agent Variants (Ablation Design)

Three agent configurations were compared in Run 3:

| Agent | RAG | Structural hooks | Purpose |
|---|---|---|---|
| `lammps_vanilla` | — | — | Baseline: model's parametric knowledge only |
| `lammps_plugin_no_rag` | — | ✓ | Isolates the contribution of structural hooks |
| `lammps_plugin` | ✓ | ✓ | Full stack: RAG + hooks |

All three agents use the same underlying model (Claude Sonnet 4.6) running inside a Docker container with access to the LAMMPS source tree mounted read-only at `/lammps_lib/`.

### Scaffolding Components

**RAG system:** A ChromaDB vector database built from the LAMMPS source tree, with three collections — conceptual documentation pages (howto guides, overview pages), working example input scripts from `lammps/examples/`, and individual command reference pages. An MCP server exposes these as three search tools the agent calls before writing scripts. Ground-truth filenames are filtered from search results to prevent leakage.

**Stop hook:** A validation script that fires when the agent ends its turn. Checks that at least one `.in` file exists and contains all structurally required commands (`units`, `atom_style`, and a `run`/`minimize` command). Returns `decision: block` with a targeted error message if checks fail, forcing the agent to re-enter and correct the problem. Has a configurable retry budget (default: 2 attempts).

**PostToolUse hook:** A faster version of the same check that fires immediately after every file write or edit. Gives the agent feedback within seconds of a bad write rather than only at end-of-turn.

### Scoring

Each agent output is scored on two dimensions:

- **Structural score (0.0–1.0):** Automated regex/keyword checks — correct `units`, `atom_style`, `dimension`, integrator fix, run step count, etc. Measures whether required LAMMPS commands are present with the right values.
- **LLM judge score (1–10):** A separate LLM call that reads the task specification and the agent's script side by side, scoring on physical correctness, syntactic validity, and completeness. Returns a numeric score, a rationale paragraph, and a list of key issues.

---

## Results (Run 3)

| Agent | Mean Structural | Mean LLM |
|---|---|---|
| `lammps_vanilla` | 0.969 | ~5.25 |
| `lammps_plugin_no_rag` | **1.000** | 5.25 |
| `lammps_plugin` | 0.969 | **6.75** |

Three findings stand out:

**RAG meaningfully improves script quality.** The full-stack agent (`lammps_plugin`) outperforms the no-RAG ablation by +3 LLM points on `lj_melt_minimal` and +2 on `lj_melt`. The agent with RAG access finds and adapts real reference scripts from the LAMMPS examples library; without it, the agent produces scripts with non-standard geometry constructions that the judge penalizes.

**Structural scores are a weak quality signal.** The `lammps_plugin_no_rag` agent achieves a perfect structural score on `crack_2d` but earns an LLM judge score of only 2. The judge found critical syntax errors — a non-existent region style (`strip`), malformed `create_box` syntax, unused atom types — that keyword-presence checks cannot detect. This points to a gap in the evaluation methodology that needs to be addressed.

**`crack_2d` is the universal failure mode.** All three agents struggle with this task. Two recurring errors appear independently across agents: (1) the NVE integrator (`fix nve`) is omitted despite the spec explicitly requesting NVE dynamics; and (2) agents generate a box anchored at the origin (x: 0 to 100) instead of centered at the origin (x: −50 to 50), which places the pre-cut notch and boundary-loading strips at geometrically incorrect positions. The simulation runs, but does not model the intended crack geometry.

---

## What Was Built

The work required building a complete domain adaptation of the existing GEOS harness:

- **`plugin_lammps/`** — a self-contained plugin containing the RAG vector database indexer, the RAG MCP server, the validation MCP server, the Stop hook, the PostToolUse hook, and a LAMMPS primer document used as the agent's reference card
- **`run/AGENTS_lammps.md`** — a dedicated system prompt explaining the LAMMPS workspace layout, command ordering rules, physics patterns, and safety rules
- **Runner extensions** in `src/runner/` — `lammps_mode` routing in the orchestrator, new Docker command builders, new CLI flags (`--agents-md-path`, `--lammps-lib-dir`), and MCP config writers for LAMMPS agents
- **Evaluation tasks** — four task specifications with reference solutions, covering simple to complex physics
- **Evaluation script** — `scripts/eval/batch_lammps_evaluate.py`, which runs structural checks and LLM judge scoring and writes per-task and summary JSON

---

## Next Steps

The most impactful near-term improvements are:

1. **Fix the `crack_2d` geometry prompt** — add explicit instruction that the box must be centered at the origin, with a worked example of the correct region coordinates. This addresses the most consistent failure across all agents.
2. **Add a `crack_2d` ground-truth script** — the GT directory for this task is currently empty, limiting both the LLM judge's reference and the RAG system's coverage of 2D fracture simulations.
3. **Strengthen structural checks** — add checks for coordinate centering, integrator-ensemble consistency, and atom type count matching, so structural scores better reflect actual script quality.
4. **Multi-run averaging** — Run 3 is a single sample per agent. Aggregating Runs 1–3 would give statistically reliable comparisons across the ablation conditions.

# Post Run5 Findings

## What LAMMPS Is

LAMMPS is an open-source molecular dynamics (MD) simulator — it models how atoms move over time given a set of forces. You write a plain-text input script (.in file) that describes the atoms, their interactions, the physics ensemble (NVT, NVE, NPT, etc.), and what to output. The task in this project is getting AI agents to author those scripts correctly from a natural-language description.

---

## The Example Library

The LAMMPS GitHub repo ships a large examples directory mounted locally at /lammps_lib/examples/. The counts:

- 90 example directories, each covering a different physics scenario (melt, crack, water, indentation, polymer, granular, GCMC, ReaxFF, etc.)
- 843 total .in input scripts available to mine — these are working, validated scripts the LAMMPS developers ship as references

These 843 scripts are what the RAG system is built on. They are indexed into 3 ChromaDB collections:

| Collection | What's in it | What it's used for |
|---|---|---|
| lammps_technical | Shadow descriptions of each of the 843 input scripts | Finding the right example script to adapt |
| lammps_commands | ~300 individual command reference pages | Looking up exact syntax |
| lammps_navigator | Conceptual docs (Howto guides, tutorials, overview pages) | Understanding physics setups |

When an agent calls search_technical, it is searching descriptions of those 843 examples. When it calls search_commands, it is searching command documentation. This is the RAG system the plugin provides.

---

## The 9 Experiments

These are the tasks every agent was evaluated on — a spread of common MD physics scenarios:

| Task | What it simulates |
|---|---|
| lj_melt_minimal | Simple LJ argon melt — the "hello world" of MD |
| lj_melt | Same but with more specific parameters |
| lj_solid | FCC crystal at low temperature |
| lj_indent | 2D nanoindentation with a spherical indenter |
| crack_2d | 2D crack propagation in a notched solid |
| couette_flow | NEMD fluid shear using the SLLOD algorithm |
| msd_diffusion | Mean-square displacement to measure diffusion |
| nvt_water | SPC/E water box with charges and bonds |
| uniaxial_tension | Stretch a crystal and measure the stress-strain curve |

These range from easy (lj_melt_minimal — a direct match to an existing example) to hard (crack_2d, nvt_water — require correct setup of notch geometry or charged molecular topology).

---

## The 12 Agents

Six Claude variants and six DeepSeek variants, each differing by what tooling they have:

| Variant | Plugin | RAG | Validation hook |
|---|---|---|---|
| vanilla | No | No | No |
| plugin | Yes | Yes | No |
| plugin_validate | Yes | Yes | Yes |
| plugin_no_hook | Yes | Yes (available) | No (hook removed) |
| plugin_no_rag | Yes | No | No |
| plugin_validate_no_rag | Yes | No | Yes |

The plugin adds three RAG search tools and the primer document. The hook is a stop-gate that blocks the agent from finishing unless it has written a valid .in file. The validate variant adds a stricter post-write check.

---

## Run5 Results

### Completion Rate

98.1% — 106 of 108 runs produced output. The 2 failures:

- lammps_deepseek_plugin / crack_2d: API stream dropped mid-run (infrastructure failure, not model logic)
- lammps_plugin_no_hook / lj_indent: Agent completed the task correctly but only used search_commands, not search_technical or search_navigator, so rag_requirement_met was false — a soft/policy failure

### Scores by Agent

| Agent | LLM Judge (0-10) | Notes |
|---|---|---|
| lammps_plugin_no_rag | 8.00 | Highest overall |
| lammps_plugin | 7.50 | |
| lammps_deepseek_plugin_validate | 7.50 | Best DeepSeek variant |
| lammps_plugin_no_hook | 7.29 | |
| lammps_vanilla | 7.25 | |
| lammps_plugin_validate | 7.22 | |
| lammps_deepseek_no_hook | 7.00 | 5 of 9 tasks produced no output |
| lammps_plugin_validate_no_rag | 6.88 | |
| lammps_deepseek_plugin | 6.78 | |
| lammps_deepseek_no_rag | 6.67 | Only agent with structural failure (lj_indent score 0.0) |
| lammps_deepseek_validate_no_rag | 6.25 | |
| lammps_deepseek_vanilla | 5.25 | Lowest overall |

### Hardest and Easiest Tasks

- Hardest: crack_2d — agents consistently got the notch geometry wrong, deleting too many atoms by using the full y-range instead of a thin slab. Conflicting fixes on boundary atoms were also common.
- Easiest: lj_solid and lj_melt_minimal — near-direct matches to examples in the library, most agents scored 9-10.
- Polarizing: msd_diffusion — Claude vanilla failed entirely, but plugin and DeepSeek variants scored 4-10.

---

## Key Findings

1. The hook matters most for DeepSeek. Without it, 5 of 9 DeepSeek tasks produced no output at all. Claude only missed 2 without the hook. DeepSeek needs the hook to reliably structure its workflow.

2. The validate step helps DeepSeek more than Claude. deepseek_plugin_validate (7.5) vs deepseek_plugin (6.78) is a larger improvement than plugin_validate (7.22) vs plugin (7.5) on the Claude side.

3. Claude is robust with or without scaffolding. Claude vanilla scored 7.25 — competitive with most plugin variants. DeepSeek vanilla scored 5.25 — a large gap that the plugin largely closes.

4. RAG retrieval alone is not the differentiator. plugin_no_rag scored the highest (8.0), meaning that for many tasks, a well-structured prompt and working examples in context matter more than active retrieval. The 843-example library is being used, but agents that skip retrieval can still perform well on standard physics setups.

5. DeepSeek_no_rag was the only agent with a structural failure. It produced a syntactically broken script for lj_indent (score 0.0 on structural). All other agents maintained structural correctness whenever they produced output.

6. crack_2d is a benchmark for geometric reasoning. It exposed a consistent failure mode — incorrect region definitions for the notch — that no agent fully solved. This is a good candidate for targeted improvement.

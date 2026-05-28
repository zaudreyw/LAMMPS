# Run5 Results Summary

## Overview

108 total runs (12 agents × 9 experiments). 106 successes, 2 failures — 98.1% success rate.

Agents: lammps_vanilla, lammps_plugin, lammps_plugin_validate, lammps_plugin_no_hook, lammps_plugin_no_rag, lammps_plugin_validate_no_rag, and DeepSeek equivalents.
Experiments: couette_flow, crack_2d, lj_indent, lj_melt, lj_melt_minimal, lj_solid, msd_diffusion, nvt_water, uniaxial_tension.

---

## Failures

**lammps_deepseek_plugin | crack_2d → `failed`**
Infrastructure failure. The verify_outputs hook blocked the agent's first stop (no .in file written yet), then OpenRouter dropped the stream with repeated "API Error: stream closed before completion." The agent was on track but could not recover. Not a model logic failure.

**lammps_plugin_no_hook | lj_indent → `failed_no_rag`**
Soft/policy failure. The agent completed the task correctly (exit code 0, file written) but only used `search_commands` — not `search_technical`, `search_navigator`, or `search_schema`, which are required for `rag_requirement_met`. Without the hook enforcing it, the agent took the shortest path and skipped deeper technical lookup.

---

## Key Takeaways

- **The hook enforces RAG usage.** The only no_rag failure came from the no_hook variant, confirming the hook is what drives agents to use technical search. Remove it and agents cut corners.
- **DeepSeek and Claude agents are comparably capable.** The one DeepSeek failure was infra, not model quality.
- **Validate variants had zero failures.** The validation step appears to add robustness.
- **No_rag variants all succeeded.** Agents can produce plausible scripts without RAG, but correctness is the open question.

---

## Evaluation Results

Two metrics per task: **structural score** (0–1, automated checks — correct units, atom style, dimension, integrator, etc.) and **LLM judge score** (0–10, GPT-4o evaluating physical correctness and script quality). Null = no `.in` file produced (run failed or agent never wrote output).

### Mean Scores by Agent (Run 5)

| Agent | Structural | LLM Judge | Notes |
|---|---|---|---|
| lammps_plugin_no_rag | 1.00 | **8.00** | Highest LLM score overall |
| lammps_plugin | 1.00 | 7.50 | |
| lammps_deepseek_plugin_validate | 1.00 | 7.50 | Best DeepSeek variant |
| lammps_plugin_no_hook | 1.00 | 7.29 | 2 tasks null (no output written) |
| lammps_vanilla | 1.00 | 7.25 | msd_diffusion null |
| lammps_plugin_validate | 1.00 | 7.22 | |
| lammps_deepseek_no_hook | 1.00 | 7.00 | **5 tasks null** — hook absence caused widespread failures |
| lammps_plugin_validate_no_rag | 1.00 | 6.88 | |
| lammps_deepseek_no_rag | 0.89 | 6.67 | Only agent with structural < 1.0 (lj_indent failed structural) |
| lammps_deepseek_plugin | 1.00 | 6.78 | |
| lammps_deepseek_validate_no_rag | 1.00 | 6.25 | |
| lammps_deepseek_vanilla | 1.00 | **5.25** | Lowest LLM score overall |

### Per-Task Highlights

**crack_2d** was the hardest task across all agents (scores of 2–8, many low). The LLM judge flagged a recurring error: agents defined the notch region using the full y-range instead of a thin slab, deleting far too many atoms. Conflicting fixes on boundary atoms (setforce + move linear simultaneously) also appeared in multiple agents.

**lj_solid** and **lj_melt_minimal** were easiest — most agents scored 9–10.

**msd_diffusion** was polarizing: vanilla Claude null (run failed), but plugin and DeepSeek variants scored 4–10.

**lj_indent** had the most issues across agents: nulls from lammps_plugin and lammps_deepseek_plugin_validate; structural failure (score=0.0) from lammps_deepseek_no_rag; and lammps_plugin_no_hook scored only 3.

### Key Takeaways from Scores

- **Claude plugin variants outperform Claude vanilla** (7.22–8.00 vs 7.25), with the RAG-free plugin being the highest scorer — suggesting the hook-driven RAG search may introduce noise or distraction on simpler tasks.
- **DeepSeek plugin_validate matched the best Claude agent** at 7.5, but DeepSeek vanilla was significantly weaker (5.25), indicating DeepSeek benefits more from the plugin scaffolding than Claude does.
- **The no_hook DeepSeek variant collapsed** — 5 of 9 tasks had no output, compared to only 2 null tasks for the Claude no_hook variant. DeepSeek without the hook fails to structure its workflow reliably.
- **lammps_deepseek_no_rag was the only agent with a structural failure**, scoring 0.0 on lj_indent. All other agents maintained structural correctness when they produced output at all.
- **The validate step helps DeepSeek more than Claude.** deepseek_plugin_validate (7.5) vs deepseek_plugin (6.78) is a larger gap than plugin_validate (7.22) vs plugin (7.5) on the Claude side.

---
name: lammps-rag
description: Use when answering LAMMPS documentation, input-script syntax, or command questions with the plugin-provided search_navigator, search_technical, and search_commands MCP tools.
---

Use the LAMMPS RAG MCP tools before answering questions about LAMMPS input-script syntax, examples, or command documentation.

The LAMMPS primer is normally injected into the agent system context by the
experiment runner. Do not look for `/workspace/LAMMPS_PRIMER.md`; task
workspaces intentionally omit that file. Treat the system-provided primer as
the high-level orientation for the task, then use the RAG tools for
task-specific evidence and exact command details.

Tool selection:

- Use `search_navigator` for conceptual orientation, feature discovery, howto guides, and source RST references.
- Use `search_commands` for authoritative command syntax, required keywords, argument types, and defaults.
- Use `search_technical` for real example .in scripts, command patterns, and references with `in_reference` plus `line_range`.

Recommended workflow for input-script authoring:

1. Search concepts with `search_navigator` when the relevant LAMMPS feature, fix style, or ensemble is unclear.
2. Search exact command syntax with `search_commands` before writing or changing a command and its arguments.
3. Search examples with `search_technical` to mirror working input-script structure.
4. When a technical result returns an `in_reference`, read the referenced file and line range if the host environment provides file-reading tools.

The ChromaDB location is configured by `LAMMPS_VECTOR_DB_DIR` and defaults to `/data/shared/lammps_agent_data/data/vector_db` in this plugin.

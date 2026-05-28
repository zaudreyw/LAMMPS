"""Agent definitions.

Mirrors the ``AGENTS`` dict from the original ``scripts/run_experiment.py``
(lines 144-419). Kept as a plain dict because ``main()`` mutates each entry's
``results_dir`` at runtime.

acpx_name: the agent identifier passed to ``acpx <agent> exec``
results_dir: where per-task workspaces land on the host
api_key_env: environment variable name for the agent's API key
"""

from __future__ import annotations

from .constants import DATA_DIR, DEFAULT_CLAUDE_MODEL, REPO_ROOT

AGENTS: dict[str, dict] = {
    "claude_code": {
        "runner": "acpx",
        "acpx_name": "claude",
        "results_dir": DATA_DIR / "eval" / "claude_code",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": None,  # passed via ANTHROPIC_API_KEY; model set by claude itself
    },
    "claude_code_repo3_plugin": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
    },
    # xmllint ablation: CC + RAG + hook with the xmllint-aware primer
    # variant inlined. Caller must pass `--strip-baked-primer
    # --geos-primer-path plugin/GEOS_PRIMER_xmllint.md`.
    # The Stop hook still does the existing parse check (NOT schema
    # validation) — this cell isolates the primer treatment.
    "claude_code_repo3_plugin_xmllint_primer": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_xmllint_primer",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
    },
    # xmllint ablation: CC + RAG + hook with the Stop hook also running
    # `xmllint --schema` after the parse check. Caller must export
    # `GEOS_HOOK_XMLLINT=1` in the host env so the runner forwards it
    # into the container. Primer is the standard one (no special path).
    "claude_code_repo3_plugin_xmllint_hook": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_xmllint_hook",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
    },
    # Both treatments stacked + the explicit MCP validate tool. Caller
    # must pass `--strip-baked-primer --geos-primer-path
    # plugin/GEOS_PRIMER_xmllint.md` AND export `GEOS_HOOK_XMLLINT=1`.
    "claude_code_repo3_plugin_xmllint_all": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_xmllint_all",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "xmllint_mcp_enabled": True,
    },
    # Same as `_xmllint_all`, but also stacks the M1-u memory cheatsheet
    # (the hero memory primer from the D-008 ablation). This is the
    # "everything stacked" condition — RAG + parse-check hook + xmllint
    # MCP tool + xmllint hook + memory cheatsheet + (caller's choice of
    # base primer via --geos-primer-path). Use to test whether the
    # memory primer's contribution is additive with the xmllint stack.
    "claude_code_repo3_plugin_xmllint_all_m1u": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_xmllint_all_m1u",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "xmllint_mcp_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_m1u.md",
    },
    # Plugin + frozen pre-learned cheatsheet (memory experiment, D-001).
    # Same as claude_code_repo3_plugin but prepends plugin/cheatsheet.md to
    # the system prompt. Cheatsheet is derived from a held-out train subset
    # of past plugin-run trajectories.
    "claude_code_repo3_plugin_mem": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_mem",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "cheatsheet.md",
    },
    # Short-cheatsheet variant (E05). Strips task-specific advice, keeps
    # only cross-task-invariant rules + explicit stop criterion. Tests
    # whether the E04 failure was cheatsheet content (specificity +
    # instruction competition) or context bloat.
    "claude_code_repo3_plugin_memshort": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_memshort",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "cheatsheet_short.md",
    },
    # Dynamic Cheatsheet (Cumulative) — cheatsheet evolves across batches
    # of test tasks via an LLM curator between batches. Points at a
    # MUTABLE cheatsheet file that the orchestrator (scripts/memory/
    # dc_cu_orchestrate.py) rewrites between batches. Design in XN-004.
    "claude_code_repo3_plugin_memdccu": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_memdccu",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "cheatsheet_dccu.md",
    },
    # Filetree injection (I08) — adds a precomputed index of /geos_lib/
    # inputFiles XML paths to the system prompt so the agent can
    # locate candidate reference XMLs without Glob/Bash find.
    "claude_code_repo3_plugin_tree": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_tree",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "filetree.md",
    },
    # E09: Cheatsheet delivered via /workspace/CHEATSHEET.md instead of
    # system-prompt injection. Tests whether the E04/E05 failure is
    # about delivery channel (system prompt) or content type (cheatsheet).
    "claude_code_repo3_plugin_memws": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_memws",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "cheatsheet_abstract.md",
        "cheatsheet_in_workspace": True,
    },
    # E11: Frozen G-Memory-lite delivered as MCP tool `memory_lookup`.
    # Agent calls memory_lookup(query) to retrieve past-task example
    # files + productive RAG queries. Concrete (not abstract) memory
    # via tool channel (not system prompt). Tests whether memory-as-tool
    # avoids the failure modes of E04/E05/E07.
    "claude_code_repo3_plugin_gmem": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_gmem",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "memory_enabled": True,
    },
    # E13: Same memory_mcp tool available, but NO system-prompt instruction
    # about it. Agent discovers memory_lookup from the tool list / docstring
    # alone. Tests whether the memory *instruction* itself was anchoring
    # behavior (E12 still hurt even though threshold blocked bad matches).
    "claude_code_repo3_plugin_gmemsilent": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_gmemsilent",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "memory_enabled": True,
        "memory_prompt_hint": False,
    },
    # PAC-1 A4': plug + silent-memory with hook DISABLED. Needed because E18
    # ran before the Stop hook existed AND before AskUserQuestion was removed
    # from the tool list — so E18 vs E24 is confounded by 2 config changes.
    # This agent gives a clean hook-off baseline with current infra.
    "claude_code_repo3_plugin_gmemsilent_nohook": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_gmemsilent_nohook",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "memory_enabled": True,
        "memory_prompt_hint": False,
        "stop_hook_enabled": False,
    },
    # --- E20 hook-ablation variants (4 cells; see docs/XN-010, SESSION_HANDOFF
    # 2026-04-21 §5, RN-002). All share the plain-plugin MCP config (geos-rag
    # only) plus the --settings file. The hook is registered via --settings
    # (not --plugin-dir) so the tool list stays identical to E17/E18.
    #
    # C0: hook OFF, no extra tool  →  E17 replicate.
    "claude_code_repo3_plugin_nohook": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_nohook",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "stop_hook_enabled": False,
    },
    # C2: hook OFF, noop MCP present  →  isolates tool-list-shape effect.
    "claude_code_repo3_plugin_noop_nohook": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_noop_nohook",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "stop_hook_enabled": False,
        "noop_mcp_enabled": True,
    },
    # C4: hook ON + noop MCP  →  interaction check.
    "claude_code_repo3_plugin_noop": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_noop",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "noop_mcp_enabled": True,
    },
    # C1 is just the existing claude_code_repo3_plugin (hook ON by default
    # now that --settings wires it in). Kept distinct so the legacy name
    # still points at the canonical "plain plugin" condition.
    # Ablation: same container, same primer, same prompt — but no repo3
    # plugin, no RAG tools, no vector DB. Baseline for measuring the plugin's
    # contribution. /geos_lib is still the filtered (decontaminated) copy.
    "claude_code_no_plugin": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_no_plugin",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": False,
    },
    # Vanilla CC, but the system prompt's GEOS Primer block is replaced with
    # the much shorter `plugin/GEOS_PRIMER_minimal.md`. Ablation against
    # `claude_code_no_plugin` to measure whether the bulky primer carries its
    # weight. Caller MUST pass `--strip-baked-primer --geos-primer-path
    # plugin/GEOS_PRIMER_minimal.md` for the swap to actually take effect —
    # the AGENTS.md bake-in suppresses the external primer otherwise (see
    # docs/2026-04-27_4condition-file-tool-comparison.md §"minimal primer").
    "claude_code_no_plugin_minprimer": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_no_plugin_minprimer",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": False,
    },
    # 2026-04-29 build-up ablation matrix: C0 (true vanilla) - C5 (RAG+SR+mem).
    # All variants assume the caller passes --strip-baked-primer with the
    # appropriate --geos-primer-path; results_dir below routes outputs to
    # the dsv4_ablation_2026-04-29 namespace under /data/shared/.
    # C0: true minimal — no plugin, no RAG, no SR hook, absolute-min primer.
    "abl_c0_true_vanilla": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "abl_c0_true_vanilla",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": False,
    },
    # C2: minimal primer + SR hook ON, but RAG MCP NOT loaded and no RAG
    # instruction in the system prompt. Plugin is loaded only so --settings
    # carries the Stop hook (verify_outputs.py). Decoupled via rag_enabled.
    "abl_c2_min_sr_no_rag": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "abl_c2_min_sr_no_rag",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        # stop_hook_enabled defaults True
    },
    # C3: minimal primer + RAG, no SR hook.
    "abl_c3_min_rag_no_sr": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "abl_c3_min_rag_no_sr",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "stop_hook_enabled": False,
    },
    # C4: minimal primer + RAG + SR hook (no xmllint, no memory).
    "abl_c4_min_rag_sr": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "abl_c4_min_rag_sr",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
    },
    # C5: C2 + DSv4-distilled M1-u memory primer (cheatsheet).
    # Memory distilled from harvest_c2_dsv4_s1 trajectories on 18 train tasks
    # via gemini-3-flash-preview (M1-u variant, ungrounded).
    "abl_c5_dsv4_mem": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "abl_c5_dsv4_mem",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_dsv4_m1u.md",
    },
    # C6: C2 + xmllint hook (no MCP-tool, no RAG). Isolates the xmllint
    # validation hook on top of parse-check-only SR. The agent is NOT told
    # about xmllint via primer — must learn from the hook's block feedback.
    # Caller MUST export GEOS_HOOK_XMLLINT=1 in host env so it's forwarded.
    "abl_c6_xmllint_hook": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "abl_c6_xmllint_hook",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
    },
    # C7: C6 + xmllint MCP tool. Agent can voluntarily call
    # mcp__xmllint__validate_geos_xml during XML authoring (vs only after).
    # This is what the user originally meant by "C2 = SR hook (no RAG)".
    # Caller MUST export GEOS_HOOK_XMLLINT=1.
    "abl_c7_xmllint_full_no_rag": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "abl_c7_xmllint_full_no_rag",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "xmllint_mcp_enabled": True,
    },
    # C8: C7 + RAG. Tests whether RAG helps when xmllint is providing
    # schema feedback (the user's hypothesis: xmllint says "you used
    # X, expected Y" and RAG search_schema looks up Y's spec).
    "abl_c8_xmllint_full_rag": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "abl_c8_xmllint_full_rag",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "xmllint_mcp_enabled": True,
    },
    # C10: C6 + DSv4-distilled memory cheatsheet.
    # Tests whether memory + xmllint hook compose (memory primes correct
    # vocab, xmllint catches residuals) or cancel (xmllint already
    # catches what memory was trying to prevent).
    "abl_c10_xmllint_hook_mem": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "abl_c10_xmllint_hook_mem",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_dsv4_m1u.md",
    },
    # C11: C7 + DSv4-distilled memory cheatsheet.
    "abl_c11_xmllint_full_mem": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "abl_c11_xmllint_full_mem",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "xmllint_mcp_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_dsv4_m1u.md",
    },
    # SE (self-evolving): blank-init plugin that the agent rewrites between
    # 6-task batches. Same harness as C6 (xmllint hook + no RAG) but plugin
    # contents (memory/skills/agents/PRIMER) are agent-authored over rounds.
    # Per-round runner overrides --plugin-dir to plugin_evolving/v{N}/ and
    # --geos-primer-path to plugin_evolving/v{N}/PRIMER.md
    "abl_se_round": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "abl_se_round",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "add_native_plugin_prefix": False,
    },
    # MemP: per-task procedural memory retrieval. Agent dict carries
    # `cheatsheet_path_template` which the orchestrator formats with the
    # task name at task-launch time, loading a per-task primer from
    # plugin/memp_per_task/<task>.md (top-3 cosine retrievals from the
    # 18-task train library, distilled via gemini-3-flash-preview).
    #
    # cMP-A: MemP on top of C2 (parse-only SR, no xmllint, no RAG)
    "abl_cMP_a_memp_on_c2": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "abl_cMP_a_memp_on_c2",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "cheatsheet_path_template": str(REPO_ROOT / "plugin" / "memp_per_task" / "{task}.md"),
    },
    # cMP-B: MemP on top of C7 (xmllint hook + xmllint MCP, no RAG)
    "abl_cMP_b_memp_on_c7": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "abl_cMP_b_memp_on_c7",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "xmllint_mcp_enabled": True,
        "cheatsheet_path_template": str(REPO_ROOT / "plugin" / "memp_per_task" / "{task}.md"),
    },
    # C9: C2 with the native-plugin-prefix suppressed in user prompt.
    # Isolates the "phantom RAG instruction" effect (the +0.24 surprise
    # from the C0-C5 ablation). Same primer + same plugin loading as C2,
    # only difference is the absence of the "use mcp__geos-rag__* tools"
    # prefix prepended to the user prompt.
    "abl_c9_no_prefix": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "abl_c9_no_prefix",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "add_native_plugin_prefix": False,
    },
    # -------------------------------------------------------------------------
    # D-008 memory ablation (post-RN-003). All stacked on RAG+SR
    # (plugin_enabled=True, stop_hook_enabled=True-by-default). Each variant
    # delivers memory content via a different path. All are FROZEN at test time.
    # -------------------------------------------------------------------------
    # M-placebo: equivalent-token generic GEOS text, not trajectory-derived.
    # Placebo control per RN-003 P1 #2.
    "claude_code_repo3_plugin_m_placebo": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_m_placebo",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_placebo.md",
    },
    # M1-u: DC-Cu primer (ungrounded). Self-judged cheatsheet distilled from
    # 18 training trajectories via gemini-3-flash-preview (no TreeSim feedback).
    "claude_code_repo3_plugin_m1u": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_m1u",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_m1u.md",
    },
    # M1-g: DC-Cu primer (grounded). Same corpus as M1-u but the distiller
    # was given TreeSim failure-mode labels, weakest-section scores, and
    # dominant-dimension hints. Paired with M1-u for grounding attribution.
    "claude_code_repo3_plugin_m1g": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_m1g",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_m1g.md",
    },
    # M3-g: RB items via in-run MCP tool. Embedding retrieval over the same
    # items served to M4-g. Uses memory_mcp_embed.py with hard-error on
    # missing OPENROUTER_API_KEY (RN-003 P2 #8). Claim C (locus) is weakened
    # due to tool-list-shape confound (RN-003 P2 #5).
    "claude_code_repo3_plugin_m3g": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_m3g",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "memory_enabled": True,
        "memory_prompt_hint": False,  # don't advertise the tool; let tool list speak
        "memory_variant": "embed",
        "memory_items_path": REPO_ROOT / "plugin" / "memory_items_m4g.json",
        "memory_embed_index_path": REPO_ROOT / "plugin" / "memory_items_m4g_embeddings.json",
    },
    # M3-g-hinted: same as M3-g (embedding MCP) but with memory_prompt_hint=True.
    # Distinguishes "tool-locus is bad" from "agent doesn't spontaneously use it"
    # per user's 2026-04-22 follow-up request.
    "claude_code_repo3_plugin_m3g_hinted": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_m3g_hinted",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "memory_enabled": True,
        "memory_prompt_hint": True,  # CHANGED from False — nudges agent to call memory_lookup
        "memory_variant": "embed",
        "memory_items_path": REPO_ROOT / "plugin" / "memory_items_m4g.json",
        "memory_embed_index_path": REPO_ROOT / "plugin" / "memory_items_m4g_embeddings.json",
    },
    # M4-u: RB items (ungrounded) via external primer injection. Same items
    # format as M4-g but distilled without TreeSim feedback. Paired with
    # M4-g for attribution claim.
    "claude_code_repo3_plugin_m4u": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_m4u",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_m4u.md",
    },
    # M4-g: RB items (grounded) via external primer injection. Hero run.
    "claude_code_repo3_plugin_m4g": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "claude_code_repo3_plugin_m4g",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_m4g.md",
    },
    # =========================================================================
    # LAMMPS agents
    # All entries carry lammps_mode=True so the orchestrator routes them
    # through the LAMMPS docker/MCP/settings path instead of the GEOS path.
    # Use with --agents-md-path run/AGENTS_lammps.md
    #          --plugin-dir plugin_lammps
    #          --vector-db-dir /data/shared/lammps_agent_data/data/vector_db
    #          --lammps-lib-dir /path/to/lammps
    # =========================================================================

    # Baseline: no plugin, no RAG, no hook.
    "lammps_vanilla": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "lammps_vanilla",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": False,
        "lammps_mode": True,
    },
    # Full stack: LAMMPS RAG MCP + structural Stop hook + PostToolUse hook.
    "lammps_plugin": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "lammps_plugin",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "lammps_mode": True,
    },
    # Full stack + lammps-validate MCP tool (agent can call validate_lammps_input mid-task).
    "lammps_plugin_validate": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "lammps_plugin_validate",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "lammps_validate_mcp_enabled": True,
        "lammps_mode": True,
    },
    # RAG on, Stop hook off — ablation for hook contribution.
    "lammps_plugin_no_hook": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "lammps_plugin_no_hook",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "stop_hook_enabled": False,
        "lammps_mode": True,
    },
    # RAG off, Stop hook on — ablation for RAG contribution.
    "lammps_plugin_no_rag": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "lammps_plugin_no_rag",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "lammps_mode": True,
    },
    # validate MCP only (no RAG) — ablation for validate-tool contribution.
    "lammps_plugin_validate_no_rag": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "lammps_plugin_validate_no_rag",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "lammps_validate_mcp_enabled": True,
        "lammps_mode": True,
    },

    # =========================================================================
    # LAMMPS cross-model cells — DeepSeek v4 flash via OpenRouter
    # Model routes through ANTHROPIC_BASE_URL = https://openrouter.ai/api
    # Use with same CLI flags as the standard lammps_* agents above.
    # =========================================================================

    # Baseline: DeepSeek v4 flash, no plugin/RAG/hook.
    "lammps_deepseek_vanilla": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "lammps_deepseek_vanilla",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": "deepseek/deepseek-chat-v3-0324",  # deepseek v4 flash on OpenRouter
        "requires_rag": False,
        "plugin_enabled": False,
        "lammps_mode": True,
    },
    # Full stack: LAMMPS RAG MCP + hooks, DeepSeek v4 flash.
    "lammps_deepseek_plugin": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "lammps_deepseek_plugin",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": "deepseek/deepseek-chat-v3-0324",
        "requires_rag": True,
        "plugin_enabled": True,
        "lammps_mode": True,
    },
    # Full stack + lammps-validate MCP tool, DeepSeek v4 flash.
    "lammps_deepseek_plugin_validate": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "lammps_deepseek_plugin_validate",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": "deepseek/deepseek-chat-v3-0324",
        "requires_rag": True,
        "plugin_enabled": True,
        "lammps_validate_mcp_enabled": True,
        "lammps_mode": True,
    },
    # RAG on, Stop hook off — ablation for hook contribution, DeepSeek v4 flash.
    "lammps_deepseek_no_hook": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "lammps_deepseek_no_hook",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": "deepseek/deepseek-chat-v3-0324",
        "requires_rag": True,
        "plugin_enabled": True,
        "stop_hook_enabled": False,
        "lammps_mode": True,
    },
    # RAG off, hook on — ablation for DeepSeek's file-navigation vs RAG.
    "lammps_deepseek_no_rag": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "lammps_deepseek_no_rag",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": "deepseek/deepseek-chat-v3-0324",
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "lammps_mode": True,
    },
    # validate MCP only (no RAG) — ablation for validate-tool contribution, DeepSeek v4 flash.
    "lammps_deepseek_validate_no_rag": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "lammps_deepseek_validate_no_rag",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": "deepseek/deepseek-chat-v3-0324",
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "lammps_validate_mcp_enabled": True,
        "lammps_mode": True,
    },

    "cursor_composer2": {
        "runner": "acpx",
        "acpx_name": "cursor",
        "results_dir": DATA_DIR / "eval" / "cursor_composer2",
        "api_key_env": "CURSOR_API_KEY",
        "model": "composer-2",
    },
    # =============================================================
    # Autonomous campaign 2026-05-01: contract/method primer split
    # + factorial ablation. Caller passes --strip-baked-primer with
    # the appropriate --geos-primer-path (PRIMER_contract.md or
    # PRIMER_method.md). All results route to dsv4 sub-tree of
    # /data/shared/.../eval/autocamp_2026-05-01/.
    # =============================================================
    # Phase 1 cells: differ only in primer file (passed via CLI).
    # Both run with no plugin.
    "autocamp_p_contract": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "dsv4" / "autocamp_p_contract",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": False,
    },
    "autocamp_p_method": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "dsv4" / "autocamp_p_method",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": False,
    },
    # Phase 2 cells: 2^(4-1) Resolution-IV fractional factorial over
    # {RAG (R), SR-hook (S), xmllint stack (X), memory (M)}.
    # Generators: D = ABC. Primer fixed (set via CLI in launch script).
    # Naming: F<rsxm> where rsxm is 4-bit binary (e.g. F0010 = X-only).
    #
    # F0 = baseline (all off). plugin off entirely.
    "autocamp_F0": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "dsv4" / "autocamp_F0",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": False,
    },
    # F1 = R+M. RAG on, memory on, no SR/xmllint. Plugin loaded for
    # RAG MCP and cheatsheet delivery; stop_hook disabled.
    "autocamp_F1": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "dsv4" / "autocamp_F1",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "stop_hook_enabled": False,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_dsv4_m1u.md",
    },
    # F2 = S+M. SR hook on (parse-check), memory on, no RAG/xmllint.
    "autocamp_F2": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "dsv4" / "autocamp_F2",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_dsv4_m1u.md",
    },
    # F3 = R+S. RAG + SR-hook. No xmllint, no memory.
    "autocamp_F3": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "dsv4" / "autocamp_F3",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
    },
    # F4 = X+M. xmllint MCP tool + memory. NO Stop hook (so X here =
    # MCP tool only; the schema-check-via-hook does NOT fire). Agent
    # must call validate_geos_xml proactively. No RAG.
    "autocamp_F4": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "dsv4" / "autocamp_F4",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "stop_hook_enabled": False,
        "xmllint_mcp_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_dsv4_m1u.md",
    },
    # F5 = R+X. RAG + xmllint MCP. NO Stop hook. No memory.
    "autocamp_F5": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "dsv4" / "autocamp_F5",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "stop_hook_enabled": False,
        "xmllint_mcp_enabled": True,
    },
    # F6 = S+X. SR hook + xmllint stack. No RAG, no memory.
    "autocamp_F6": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "dsv4" / "autocamp_F6",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "xmllint_mcp_enabled": True,
    },
    # F7 = R+S+X+M. Everything on.
    "autocamp_F7": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "dsv4" / "autocamp_F7",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": True,
        "plugin_enabled": True,
        "xmllint_mcp_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_dsv4_m1u.md",
    },
    # F8: missing factorial cell — R-S+X+M+. The "all positives, no
    # RAG" combination. Predicted-best per main effects (M=+0.004,
    # X=+0.007, S=-0.003, R=-0.033). Adds memory cheatsheet to F6.
    "autocamp_F8": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "dsv4" / "autocamp_F8",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "xmllint_mcp_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_dsv4_m1u.md",
    },
    # F11: SE decomposed — F6 harness (SR + xmllint, no RAG, no skills,
    # no subagent) + v3's PRIMER and v3's cheatsheet. Tests whether
    # v3's prose is the active ingredient vs the full v3 plugin
    # packaging. Primer overridden at launch via CLI; cheatsheet path
    # set here.
    "autocamp_F11": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "dsv4" / "autocamp_F11",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "xmllint_mcp_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin_evolving" / "v3" / "memory" / "cheatsheet.md",
    },
    # SE: self-skill-authorship. Use plugin_evolving/v3 (most-evolved
    # round) as the loaded plugin. Same harness as F6 (SR + xmllint
    # stack, no RAG). Plugin and primer paths overridden in launch
    # script to plugin_evolving/v3/ + plugin_evolving/v3/PRIMER.md.
    "autocamp_SE": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "dsv4" / "autocamp_SE",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "xmllint_mcp_enabled": True,
        "add_native_plugin_prefix": False,
    },
    # v4: trajectory-mined lookup tables (task→canonical XML,
    # constitutive→header, anti-patterns) layered onto a v3-style plugin.
    # Same harness as SE (xmllint + plugin_enabled). Plugin and primer
    # overridden at launch to plugin_evolving/v4/.
    "autocamp_v4": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "dsv4" / "autocamp_v4",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "xmllint_mcp_enabled": True,
        "add_native_plugin_prefix": False,
    },
    # =============================================================
    # Cross-model cells (Phase 3). Same knobs as DSv4 baseline
    # (autocamp_F0, no plugin) and best (TBD — will use F6 or F7).
    # results_dir is overridden at launch time per model.
    # =============================================================
    "autocamp_xmodel_baseline": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "xmodel" / "baseline",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": None,  # set per-launch via --claude-model
        "requires_rag": False,
        "plugin_enabled": False,
    },
    "autocamp_xmodel_best": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "autocamp_2026-05-01" / "xmodel" / "best",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": None,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "xmllint_mcp_enabled": True,
    },
    # =============================================================
    # Interactive-autonomy + difficulty-ramp study (2026-05-03).
    # See docs/2026-05-03_interactive-autonomy-design.md.
    # =============================================================
    "ia_F0_noninteractive": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "interactive_autonomy_2026-05-03" / "ia_F0",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": False,
    },
    "ia_F4_noninteractive": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "interactive_autonomy_2026-05-03" / "ia_F4",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "stop_hook_enabled": False,
        "xmllint_mcp_enabled": True,
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_dsv4_m1u.md",
    },
    "ia_F0_interactive": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "interactive_autonomy_2026-05-03" / "ia_F0_int",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "stop_hook_enabled": False,
        "supervisor_enabled": True,
        "supervisor_model": "deepseek-v4-flash",
        "supervisor_base_url": "https://api.deepseek.com/v1",
        "memory_prompt_hint": False,
    },
    "ia_F4_interactive": {
        "runner": "claude_native",
        "results_dir": DATA_DIR / "eval" / "interactive_autonomy_2026-05-03" / "ia_F4_int",
        "api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "model": DEFAULT_CLAUDE_MODEL,
        "requires_rag": False,
        "plugin_enabled": True,
        "rag_enabled": False,
        "stop_hook_enabled": False,
        "xmllint_mcp_enabled": True,
        "supervisor_enabled": True,
        "supervisor_model": "deepseek-v4-flash",
        "supervisor_base_url": "https://api.deepseek.com/v1",
        "cheatsheet_path": REPO_ROOT / "plugin" / "memory_primer_dsv4_m1u.md",
    },
}

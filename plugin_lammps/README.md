# LAMMPS Plugin

Claude Code plugin for LAMMPS input authoring. Mirrors the GEOS plugin
architecture: a 3-collection ChromaDB RAG MCP server + self-verification hooks.

## Directory structure

```
plugin_lammps/
├── .claude-plugin/
│   └── plugin.json           ← plugin manifest + MCP server config
├── scripts/
│   └── lammps_rag_mcp.py     ← RAG MCP server (3 search tools)
├── hooks/
│   ├── hooks.json            ← Stop + PostToolUse hook config
│   ├── verify_outputs.py     ← Stop hook: structural validation
│   └── verify_lammps_post_write.py  ← PostToolUse hook
├── LAMMPS_PRIMER_minimal.md  ← agent quick-reference primer
└── README.md
```

## Local testing

```bash
cd plugin_lammps
claude --plugin-dir .
```

---

## Building the ChromaDB vector database

The RAG server expects three ChromaDB collections populated from the LAMMPS
source tree. Run the indexer (to be created at `scripts/build_lammps_vector_db.py`)
pointing at a LAMMPS checkout:

```bash
uv run scripts/build_lammps_vector_db.py \
    --lammps-src /path/to/lammps \
    --vector-db-dir /data/lammps_agent_data/vector_db
```

### What to index and why

#### Collection 1 — `lammps_navigator` (conceptual docs)

**Source**: `doc/src/` RST files that describe concepts rather than individual commands.

Useful files to include:

| File pattern | Why useful |
|---|---|
| `Howto_*.rst` | Howto guides ("how do I run NVT?", "how do I use GCMC?") |
| `Tutorial_*.rst` | Step-by-step tutorials with full examples |
| `Speed_*.rst` | Performance tuning guides |
| `Packages_*.rst` | Which package provides which feature |
| `Commands_*.rst` | Overview pages for command categories |
| `atom_style.rst` | Lists all atom styles and their requirements |
| `units.rst` | Unit system reference table |
| `boundary.rst` | Boundary condition explanations |
| `pair_style.rst` | Overview of pair interaction styles |
| `fix.rst` | Overview of fix commands |
| `compute.rst` | Overview of compute commands |
| `dump.rst` | Output format options |
| `Overview_*.rst` | High-level orientation pages |
| `Intro_*.rst` | Introduction pages |
| `Errors_*.rst` | Common error messages and meanings (very useful for debugging) |

Indexing strategy: chunk by RST section heading (H1/H2), keep 300–600 tokens
per chunk, store `source_path`, `title`, `breadcrumbs`, `chunk_type` metadata.

#### Collection 2 — `lammps_technical` (example input scripts)

**Source**: `examples/` subdirectories, one per physics type.

The LAMMPS `examples/` tree is organized as:

```
examples/
  melt/         in.melt             ← classic LJ melt
  crack/        in.crack            ← fracture propagation
  COUPLE/       …                   ← coupled simulations
  GCMC/         in.gcmc             ← grand canonical MC
  pour/         in.pour             ← granular pour
  polymer/      in.polymer          ← chain molecules
  water/        in.water, in.tip4p  ← water models
  dipole/       in.dipole           ← polar molecules
  indent/       in.indent           ← nanoindentation
  shear/        in.shear            ← shear deformation
  flow/         in.flow, in.pois    ← fluid flow
  rigid/        in.rigid            ← rigid body dynamics
  neb/          in.neb              ← nudged elastic band
  tad/          in.tad              ← temperature-accelerated dynamics
  reax/         in.reax, in.crack   ← ReaxFF reactive MD
  peptide/      in.peptide          ← protein fragment
  …many more
```

Indexing strategy: for each `in.*` script, generate a "shadow" description:
- Parse the script to extract the unit system, atom style, pair style, and
  any fix styles used.
- Write a 2–4 sentence natural-language description of what the simulation does.
- Embed the shadow text (not the raw script) since the raw commands are very
  sparse and don't embed well semantically.
- Store `in_reference` (the path to the real script), `source_path`, `title`,
  `line_range`, `breadcrumbs` in metadata.

#### Collection 3 — `lammps_commands` (command reference)

**Source**: Individual command RST pages in `doc/src/`.

LAMMPS has ~300 commands. The most commonly needed ones:

| Category | Key RST files |
|---|---|
| Core workflow | `units.rst`, `atom_style.rst`, `boundary.rst`, `lattice.rst`, `region.rst`, `create_box.rst`, `create_atoms.rst`, `read_data.rst`, `read_restart.rst` |
| Pair potentials | `pair_lj_cut.rst`, `pair_eam.rst`, `pair_tersoff.rst`, `pair_reaxff.rst`, `pair_lj_cut_coul_long.rst`, `kspace_style.rst` |
| Bonds/angles | `bond_harmonic.rst`, `angle_harmonic.rst`, `dihedral_opls.rst` |
| Integrators | `fix_nve.rst`, `fix_nvt.rst`, `fix_npt.rst`, `fix_langevin.rst` |
| Constraints | `fix_rigid.rst`, `fix_shake.rst`, `fix_spring.rst` |
| Outputs | `thermo.rst`, `thermo_style.rst`, `dump.rst`, `dump_image.rst` |
| Computes | `compute_temp.rst`, `compute_msd.rst`, `compute_rdf.rst`, `compute_stress_atom.rst` |
| Special | `fix_gcmc.rst`, `fix_nh.rst`, `minimize.rst`, `run.rst`, `variable.rst`, `if.rst` |

Indexing strategy: chunk each RST file by section. For the syntax block at the
top of each page (the "Syntax" section), store it as a standalone chunk tagged
`chunk_type=syntax`. Store `command_name`, `title`, `source_path` in metadata.

### Environment variables

Set these before running the indexer or the MCP server:

```bash
export LAMMPS_VECTOR_DB_DIR=/data/lammps_agent_data/vector_db
export OPENROUTER_API_KEY=<your-key>
export LAMMPS_EMBEDDING_MODEL_NAME=qwen/qwen3-embedding-8b  # or text-embedding-3-small
```

---

## Running the harness with the LAMMPS plugin

```bash
# Build the Docker image (update run/Dockerfile to mount /lammps_lib)
docker build -t lammps-eval run/

# Run experiments
python -m src.runner.cli \
    --run lammps_exp1 \
    --agents claude_code_repo3_plugin \
    --agents-md-path run/AGENTS_lammps.md \
    --plugin-dir plugin_lammps \
    --vector-db-dir /data/lammps_agent_data/vector_db \
    --geos-primer-path plugin_lammps/LAMMPS_PRIMER_minimal.md \
    --strip-baked-primer \
    --geos-lib-dir /path/to/lammps  # mounts as /lammps_lib in container
```

## Validation

The Stop hook (`verify_outputs.py`) has three tiers:

1. **Presence**: blocks if no `.in` files exist under `/workspace/inputs/`
2. **Structure**: blocks if `units`, `atom_style`, or any `run`/`minimize`
   command is missing
3. **LAMMPS binary** (opt-in via `LAMMPS_HOOK_LAMMPS_CHECK=1`): runs the
   LAMMPS binary and blocks on non-zero exit. Disabled by default because it
   requires all referenced data files and potential files to be present.

LAMMPS does not have an equivalent of `xmllint --schema` — the binary itself
is the authoritative validator.

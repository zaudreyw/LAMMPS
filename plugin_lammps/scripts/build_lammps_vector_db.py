# /// script
# dependencies = [
#   "chromadb>=1.0.0,<2",
#   "openai>=1.0.0,<3",
#   "python-dotenv>=1.0.0,<2",
#   "tqdm>=4.0.0",
# ]
# ///
"""Build the LAMMPS ChromaDB vector database from a LAMMPS source tree.

Creates three collections in a ChromaDB persistent store:

  lammps_navigator  — RST conceptual/howto pages, chunked by section
  lammps_technical  — Example .in scripts embedded via synthetic shadow text
  lammps_commands   — Command-reference RST pages, chunked by section

Usage:
    uv run scripts/build_lammps_vector_db.py \\
        --lammps-src /path/to/lammps \\
        --vector-db-dir /data/lammps_agent_data/vector_db

    # With LLM-generated shadow descriptions (higher quality, costs API calls):
    uv run scripts/build_lammps_vector_db.py \\
        --lammps-src /path/to/lammps \\
        --vector-db-dir /data/lammps_agent_data/vector_db \\
        --shadow-mode llm

Environment variables:
    OPENROUTER_API_KEY   Required for embedding (and for --shadow-mode llm)
    LAMMPS_EMBEDDING_MODEL_NAME  Override default embedding model
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

load_dotenv()

DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
COLLECTION_NAVIGATOR = "lammps_navigator"
COLLECTION_TECHNICAL = "lammps_technical"
COLLECTION_COMMANDS = "lammps_commands"

# RST filename prefixes that go into the navigator (conceptual) collection
_NAVIGATOR_PREFIXES = (
    "Howto_", "Tutorial_", "Speed_", "Packages_", "Commands_",
    "Overview_", "Intro_", "Errors_",
)
# RST files by exact stem that go into navigator (one-off useful pages)
_NAVIGATOR_STEMS = {
    "atom_style", "units", "boundary", "pair_style", "fix", "compute",
    "dump", "run", "minimize",
}
# RST filename prefixes that go into the commands collection
_COMMANDS_PREFIXES = (
    "pair_", "bond_", "angle_", "dihedral_", "improper_", "fix_",
    "compute_", "kspace_", "dump_", "timestep", "thermo", "region",
    "lattice", "create_", "read_", "write_", "velocity", "neighbor",
    "variable", "if", "jump", "label", "next", "print", "run",
    "minimize", "units", "atom_style", "boundary", "mass", "group",
    "set", "displace_atoms", "change_box", "replicate", "reset_",
    "dynamical_matrix", "neb", "prd", "tad", "server", "temper",
)

MAX_CHUNK_TOKENS = 500   # approximate; 1 token ≈ 4 chars
CHUNK_OVERLAP_LINES = 3  # lines of context to carry across section boundaries


# ---------------------------------------------------------------------------
# RST parsing helpers
# ---------------------------------------------------------------------------

def _section_heading_level(line: str, underline: str) -> int | None:
    """Return 1 for H1, 2 for H2, None if not an RST heading underline."""
    if len(underline.strip()) < 3:
        return None
    char = underline.strip()[0]
    if not all(c == char for c in underline.strip()):
        return None
    if len(underline.strip()) < len(line.rstrip()):
        return None
    # RST heading chars by convention (not enforced by the spec, but LAMMPS docs use this order)
    order = "#*=-.~^"
    try:
        return order.index(char) + 1
    except ValueError:
        return len(order) + 1


def _split_rst_into_chunks(text: str, source_path: str, max_chars: int = 2000) -> list[dict[str, Any]]:
    """Split RST text into chunks at section boundaries."""
    lines = text.splitlines()
    chunks: list[dict[str, Any]] = []
    current_lines: list[str] = []
    current_title = Path(source_path).stem.replace("_", " ")
    breadcrumbs: list[str] = []

    def flush(title: str) -> None:
        body = "\n".join(current_lines).strip()
        if not body:
            return
        chunks.append({
            "text": body[:max_chars],
            "title": title,
            "breadcrumbs": " > ".join(breadcrumbs) if breadcrumbs else title,
            "source_path": source_path,
            "chunk_type": "section",
        })

    i = 0
    while i < len(lines):
        line = lines[i]
        # Check if next line is a heading underline
        if i + 1 < len(lines):
            underline = lines[i + 1]
            level = _section_heading_level(line, underline)
            if level is not None and level <= 3:
                flush(current_title)
                current_lines = []
                current_title = line.strip()
                if level == 1:
                    breadcrumbs = [current_title]
                elif level == 2:
                    breadcrumbs = breadcrumbs[:1] + [current_title]
                else:
                    breadcrumbs = breadcrumbs[:2] + [current_title]
                i += 2  # skip title + underline
                continue
        current_lines.append(line)
        # Auto-flush if chunk is getting large
        if sum(len(l) for l in current_lines) > max_chars:
            flush(current_title)
            current_lines = current_lines[-CHUNK_OVERLAP_LINES:]
        i += 1

    flush(current_title)
    return chunks


# ---------------------------------------------------------------------------
# Example script shadow text generation
# ---------------------------------------------------------------------------

def _extract_in_metadata(text: str) -> dict[str, str]:
    """Parse a LAMMPS .in script and extract key command values."""
    meta: dict[str, str] = {}
    pair_styles: list[str] = []
    fix_styles: list[str] = []
    for line in text.splitlines():
        stripped = line.split("#")[0].strip()
        if not stripped:
            continue
        tokens = stripped.split()
        if not tokens:
            continue
        cmd = tokens[0].lower()
        rest = " ".join(tokens[1:])
        if cmd == "units" and "units" not in meta:
            meta["units"] = rest.split()[0] if rest else ""
        elif cmd == "atom_style" and "atom_style" not in meta:
            meta["atom_style"] = rest.split()[0] if rest else ""
        elif cmd == "dimension" and "dimension" not in meta:
            meta["dimension"] = rest.split()[0] if rest else "3"
        elif cmd == "pair_style":
            pair_styles.append(rest.split()[0] if rest else rest)
        elif cmd in ("fix", "fix_modify") and len(tokens) >= 4:
            fix_styles.append(tokens[3])
        elif cmd == "run" and "run" not in meta:
            meta["run"] = rest.split()[0] if rest else ""
        elif cmd == "minimize" and "run" not in meta:
            meta["run"] = "minimize"
        elif cmd == "read_data" and "read_data" not in meta:
            meta["read_data"] = rest.split()[0] if rest else ""
        elif cmd == "create_atoms" and "create_atoms" not in meta:
            meta["create_atoms"] = rest.split()[0] if rest else ""
    meta["pair_styles"] = ", ".join(dict.fromkeys(pair_styles)) if pair_styles else "unknown"
    meta["fix_styles"] = ", ".join(dict.fromkeys(fix_styles)) if fix_styles else "none"
    return meta


def _synthetic_shadow(path: Path, text: str) -> str:
    """Generate a template-based shadow description for a LAMMPS .in script."""
    meta = _extract_in_metadata(text)
    parts: list[str] = []
    script_name = path.parent.name

    # Headline
    units = meta.get("units", "unknown")
    atom_style = meta.get("atom_style", "unknown")
    dim = meta.get("dimension", "3")
    parts.append(
        f"LAMMPS example script '{path.name}' from the '{script_name}' example directory. "
        f"Uses {units} units, {atom_style} atom style, {dim}D simulation."
    )

    pair = meta.get("pair_styles", "")
    if pair and pair != "unknown":
        parts.append(f"Pair interactions: {pair}.")

    fixes = meta.get("fix_styles", "")
    if fixes and fixes != "none":
        parts.append(f"Integrators/fixes include: {fixes}.")

    run_val = meta.get("run", "")
    if run_val == "minimize":
        parts.append("Performs energy minimization.")
    elif run_val:
        parts.append(f"Runs for {run_val} timesteps.")

    if meta.get("read_data"):
        parts.append("Reads initial geometry from a data file.")
    elif meta.get("create_atoms"):
        parts.append("Creates atoms on a lattice using create_atoms.")

    return " ".join(parts)


def _llm_shadow(path: Path, text: str, client: Any, model: str) -> str:
    """Generate a shadow description via an LLM call."""
    snippet = text[:3000]  # keep costs low
    try:
        resp = client.chat.completions.create(
            model=model,
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Write a 2-4 sentence description of what this LAMMPS input script does. "
                        "Focus on: what physics is being simulated, the unit system, atom style, "
                        "pair potential, and integrator/ensemble. Be specific and concrete.\n\n"
                        f"Script name: {path.name}\n"
                        f"Directory: {path.parent.name}\n\n"
                        f"```\n{snippet}\n```"
                    ),
                }
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        print(f"  WARN: LLM shadow failed for {path.name}: {exc}; falling back to synthetic")
        return _synthetic_shadow(path, text)


# ---------------------------------------------------------------------------
# Collection builders
# ---------------------------------------------------------------------------

def _rst_files_for_navigator(doc_src: Path) -> list[Path]:
    results: list[Path] = []
    for rst in doc_src.glob("*.rst"):
        stem = rst.stem
        if any(stem.startswith(p) for p in _NAVIGATOR_PREFIXES):
            results.append(rst)
        elif stem in _NAVIGATOR_STEMS:
            results.append(rst)
    return sorted(results)


def _rst_files_for_commands(doc_src: Path) -> list[Path]:
    results: list[Path] = []
    for rst in doc_src.glob("*.rst"):
        stem = rst.stem
        if any(stem.startswith(p) for p in _COMMANDS_PREFIXES):
            results.append(rst)
    return sorted(results)


def _in_files(examples_dir: Path) -> list[Path]:
    results: list[Path] = []
    for p in examples_dir.rglob("*"):
        if p.is_file():
            name = p.name
            # LAMMPS uses both "in.name" and "name.in" conventions
            if name.startswith("in.") or name.endswith(".in"):
                results.append(p)
    return sorted(results)


def build_navigator_collection(
    collection: Any,
    doc_src: Path,
    *,
    batch_size: int = 50,
) -> int:
    rst_files = _rst_files_for_navigator(doc_src)
    print(f"  navigator: indexing {len(rst_files)} RST files")
    total = 0
    for rst_path in tqdm(rst_files, desc="  navigator"):
        try:
            text = rst_path.read_text(errors="replace")
        except OSError:
            continue
        rel = str(rst_path.relative_to(doc_src.parent))
        chunks = _split_rst_into_chunks(text, source_path=rel)
        docs, metas, ids = [], [], []
        for idx, chunk in enumerate(chunks):
            doc_id = f"nav_{rst_path.stem}_{idx}"
            docs.append(chunk["text"])
            metas.append({
                "title": chunk["title"],
                "breadcrumbs": chunk["breadcrumbs"],
                "source_path": chunk["source_path"],
                "chunk_type": chunk["chunk_type"],
            })
            ids.append(doc_id)
        for i in range(0, len(docs), batch_size):
            collection.upsert(
                documents=docs[i:i + batch_size],
                metadatas=metas[i:i + batch_size],
                ids=ids[i:i + batch_size],
            )
        total += len(docs)
    return total


def build_commands_collection(
    collection: Any,
    doc_src: Path,
    *,
    batch_size: int = 50,
) -> int:
    rst_files = _rst_files_for_commands(doc_src)
    print(f"  commands: indexing {len(rst_files)} command RST files")
    total = 0
    for rst_path in tqdm(rst_files, desc="  commands"):
        try:
            text = rst_path.read_text(errors="replace")
        except OSError:
            continue
        rel = str(rst_path.relative_to(doc_src.parent))
        chunks = _split_rst_into_chunks(text, source_path=rel)
        docs, metas, ids = [], [], []
        for idx, chunk in enumerate(chunks):
            doc_id = f"cmd_{rst_path.stem}_{idx}"
            docs.append(chunk["text"])
            metas.append({
                "command_name": rst_path.stem,
                "title": chunk["title"],
                "source_path": chunk["source_path"],
                "chunk_type": chunk["chunk_type"],
            })
            ids.append(doc_id)
        for i in range(0, len(docs), batch_size):
            collection.upsert(
                documents=docs[i:i + batch_size],
                metadatas=metas[i:i + batch_size],
                ids=ids[i:i + batch_size],
            )
        total += len(docs)
    return total


def build_technical_collection(
    collection: Any,
    examples_dir: Path,
    lammps_src: Path,
    *,
    shadow_mode: str = "synthetic",
    llm_client: Any = None,
    llm_model: str = "",
    batch_size: int = 50,
) -> int:
    in_files = _in_files(examples_dir)
    print(f"  technical: indexing {len(in_files)} example scripts (shadow_mode={shadow_mode})")
    docs, metas, ids = [], [], []
    for in_path in tqdm(in_files, desc="  technical"):
        try:
            text = in_path.read_text(errors="replace")
        except OSError:
            continue
        if shadow_mode == "llm" and llm_client is not None:
            shadow = _llm_shadow(in_path, text, llm_client, llm_model)
        else:
            shadow = _synthetic_shadow(in_path, text)
        try:
            rel = str(in_path.relative_to(lammps_src))
        except ValueError:
            rel = str(in_path)
        doc_id = "tech_" + rel.replace("/", "_").replace("\\", "_")
        docs.append(shadow)
        metas.append({
            "title": f"{in_path.parent.name}/{in_path.name}",
            "in_reference": rel,
            "source_path": rel,
            "line_range": f"1-{text.count(chr(10)) + 1}",
            "breadcrumbs": f"examples > {in_path.parent.name} > {in_path.name}",
        })
        ids.append(doc_id)

    for i in range(0, len(docs), batch_size):
        collection.upsert(
            documents=docs[i:i + batch_size],
            metadatas=metas[i:i + batch_size],
            ids=ids[i:i + batch_size],
        )
    return len(docs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build LAMMPS ChromaDB vector database")
    parser.add_argument("--lammps-src", required=True, type=Path,
                        help="Path to LAMMPS source tree (contains doc/ and examples/)")
    parser.add_argument("--vector-db-dir", required=True, type=Path,
                        help="Output ChromaDB directory")
    parser.add_argument("--shadow-mode", choices=["synthetic", "llm"], default="synthetic",
                        help="How to generate example script descriptions (default: synthetic)")
    parser.add_argument("--llm-model", default="google/gemini-flash-1.5-8b",
                        help="Model for LLM shadow generation (only used with --shadow-mode llm)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete and recreate all collections from scratch")
    args = parser.parse_args()

    lammps_src = args.lammps_src.resolve()
    doc_src = lammps_src / "doc" / "src"
    examples_dir = lammps_src / "examples"

    if not doc_src.exists():
        print(f"ERROR: doc/src not found under {lammps_src}", file=sys.stderr)
        sys.exit(1)
    if not examples_dir.exists():
        print(f"ERROR: examples/ not found under {lammps_src}", file=sys.stderr)
        sys.exit(1)

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY (or OPENAI_API_KEY) must be set", file=sys.stderr)
        sys.exit(1)

    embedding_model = os.environ.get("LAMMPS_EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL)
    api_base = os.environ.get("OPENROUTER_API_BASE", DEFAULT_OPENROUTER_API_BASE)

    args.vector_db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(args.vector_db_dir))

    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        api_base=api_base,
        model_name=embedding_model,
    )

    llm_client = None
    if args.shadow_mode == "llm":
        from openai import OpenAI
        llm_client = OpenAI(api_key=api_key, base_url=api_base)

    print(f"LAMMPS source : {lammps_src}")
    print(f"Vector DB     : {args.vector_db_dir}")
    print(f"Embedding     : {embedding_model}")
    print(f"Shadow mode   : {args.shadow_mode}")

    def get_or_create(name: str) -> Any:
        if args.reset:
            try:
                client.delete_collection(name)
                print(f"  deleted existing collection: {name}")
            except Exception:
                pass
        return client.get_or_create_collection(name=name, embedding_function=emb_fn)

    start = time.time()
    nav_col = get_or_create(COLLECTION_NAVIGATOR)
    n_nav = build_navigator_collection(nav_col, doc_src)
    print(f"  → {n_nav} navigator chunks indexed")

    cmd_col = get_or_create(COLLECTION_COMMANDS)
    n_cmd = build_commands_collection(cmd_col, doc_src)
    print(f"  → {n_cmd} command chunks indexed")

    tech_col = get_or_create(COLLECTION_TECHNICAL)
    n_tech = build_technical_collection(
        tech_col, examples_dir, lammps_src,
        shadow_mode=args.shadow_mode,
        llm_client=llm_client,
        llm_model=args.llm_model,
    )
    print(f"  → {n_tech} technical entries indexed")

    elapsed = round(time.time() - start, 1)
    print(f"\nDone in {elapsed}s — {n_nav + n_cmd + n_tech} total chunks across 3 collections.")
    print(f"Vector DB: {args.vector_db_dir}")


if __name__ == "__main__":
    main()

# /// script
# dependencies = [
#   "chromadb>=1.0.0,<2",
#   "mcp>=1.0.0,<2",
#   "openai>=1.0.0,<3",
#   "python-dotenv>=1.0.0,<2",
# ]
# ///
"""MCP server exposing LAMMPS ChromaDB RAG search tools.

Three collections mirror the GEOS-RAG design:
  lammps_navigator  — LAMMPS RST documentation (howtos, tutorials, overview
                      pages). Useful for conceptual navigation: "which fix
                      style controls NVT?", "how do I use LAMMPS for GCMC?"
  lammps_technical  — LAMMPS example input scripts from examples/ with
                      synthetic shadow descriptions. Useful for finding
                      concrete syntax: "show me a Lennard-Jones melt example".
  lammps_commands   — Individual LAMMPS command reference pages from doc/src/.
                      Useful for exact command syntax, required keywords, and
                      flag names: "pair_style lj/cut arguments", "fix nvt tchain".

Building these collections requires a separate indexing script (see
scripts/build_lammps_vector_db.py) that reads from the LAMMPS source tree.

Environment variables:
  LAMMPS_VECTOR_DB_DIR   Path to the ChromaDB directory.
  LAMMPS_DATA_DIR        Alternative: parent of vector_db/ subdirectory.
  LAMMPS_EMBEDDING_MODEL_NAME  Override the default embedding model.
  OPENROUTER_API_KEY     API key for embedding model.
  EXCLUDED_GT_IN_FILENAMES     Comma-separated list of .in ground-truth filenames
                               to hide from RAG results (prevents GT leakage).
  EXCLUDED_RST_PATHS           Comma-separated list of .rst paths to hide.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP


PLUGIN_ROOT = Path(os.environ.get("CLAUDE_PLUGIN_ROOT", Path(__file__).resolve().parents[1]))
DEFAULT_VECTOR_DB_DIR = Path("/data/shared/lammps_agent_data/data/vector_db")
COLLECTION_NAVIGATOR = "lammps_navigator"
COLLECTION_TECHNICAL = "lammps_technical"
COLLECTION_COMMANDS  = "lammps_commands"
DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"


load_dotenv(PLUGIN_ROOT / ".env", override=False)
load_dotenv(Path.cwd() / ".env", override=False)


def _vector_db_dir() -> Path:
    explicit = os.environ.get("LAMMPS_VECTOR_DB_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()
    data_dir = os.environ.get("LAMMPS_DATA_DIR")
    if data_dir:
        return (Path(data_dir).expanduser() / "vector_db").resolve()
    return DEFAULT_VECTOR_DB_DIR


def _load_env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [item.strip() for item in raw.split(",") if item.strip()]
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return []


def _normalize_path(path: str | Path) -> str:
    return str(Path(path)).replace("\\", "/").lower()


@dataclass(frozen=True)
class ReferenceAccessPolicy:
    """Controls which files are hidden from RAG results to prevent GT leakage."""

    blocked_in_filenames: frozenset[str]   # ground-truth .in basenames
    blocked_rst_paths: frozenset[str]       # ground-truth .rst doc paths

    @classmethod
    def from_environment(cls) -> "ReferenceAccessPolicy":
        in_filenames = frozenset(
            item.lower()
            for item in _load_env_list("EXCLUDED_GT_IN_FILENAMES")
            if item.lower().endswith(".in")
        )
        return cls(
            blocked_in_filenames=in_filenames,
            blocked_rst_paths=frozenset(
                _normalize_path(item)
                for item in _load_env_list("EXCLUDED_RST_PATHS")
                if item.lower().endswith(".rst")
            ),
        )

    def is_blocked_in_path(self, path: str | Path) -> bool:
        if not self.blocked_in_filenames:
            return False
        candidate = Path(path)
        # LAMMPS example files may be named "in.something" (prefix style used
        # in the LAMMPS source) or "something.in" (suffix style used for output).
        # Block both conventions.
        name = candidate.name.lower()
        stem = candidate.stem.lower()
        suffix = candidate.suffix.lower()
        if suffix == ".in" and name in self.blocked_in_filenames:
            return True
        # Handle "in.lj" style where the blocked name is "in.lj"
        if name in self.blocked_in_filenames:
            return True
        return False

    def is_blocked_rst_path(self, path: str | Path) -> bool:
        if not self.blocked_rst_paths:
            return False
        normalized = _normalize_path(path)
        return any(
            normalized == blocked or normalized.endswith(f"/{blocked}")
            for blocked in self.blocked_rst_paths
        )


class ChromaSearchBackend:
    def __init__(self) -> None:
        self.vector_db_dir = _vector_db_dir()
        self.client = chromadb.PersistentClient(path=str(self.vector_db_dir))
        self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get(
                "OPENROUTER_API_BASE",
                os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_API_BASE),
            ),
            model_name=os.environ.get("LAMMPS_EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL),
        )
        self._collections: dict[str, Any] = {}

    def get_collection(self, name: str) -> Any:
        if name not in self._collections:
            self._collections[name] = self.client.get_collection(
                name=name,
                embedding_function=self.embedding_fn,
            )
        return self._collections[name]


mcp = FastMCP("lammps-rag")
_backend: ChromaSearchBackend | None = None
_policy: ReferenceAccessPolicy | None = None


def _get_backend() -> ChromaSearchBackend:
    global _backend
    if _backend is None:
        _backend = ChromaSearchBackend()
    return _backend


def _get_policy() -> ReferenceAccessPolicy:
    global _policy
    if _policy is None:
        _policy = ReferenceAccessPolicy.from_environment()
    return _policy


def _bounded_n_results(value: int, *, default: int, maximum: int = 20) -> int:
    try:
        n_results = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(n_results, maximum))


@mcp.tool()
def search_navigator(query: str, n_results: int = 5) -> dict[str, Any]:
    """Search LAMMPS RST documentation for conceptual navigation.

    Best for: understanding which commands or styles to use for a physics goal,
    finding howto guides ("how do I set up NVT?", "how do I use GCMC?"),
    and locating relevant documentation sections.
    """
    n_results = _bounded_n_results(n_results, default=5)
    collection = _get_backend().get_collection(COLLECTION_NAVIGATOR)
    results = collection.query(query_texts=[query], n_results=n_results)
    policy = _get_policy()

    formatted: list[dict[str, Any]] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    for index, doc in enumerate(documents):
        meta = metadatas[index]
        source_path = meta.get("source_path", "")
        if policy.is_blocked_rst_path(source_path):
            continue
        formatted.append(
            {
                "title": meta.get("title", "No Title"),
                "breadcrumbs": meta.get("breadcrumbs", ""),
                "type": meta.get("chunk_type", "unknown"),
                "source": source_path,
                "preview": doc[:200] + "..." if len(doc) > 200 else doc,
            }
        )

    return {
        "query": query,
        "results": formatted,
        "hint": (
            "Use Read with the source path under /lammps_lib/doc/src/ for full content. "
            "For example scripts, use search_technical instead."
        ),
    }


@mcp.tool()
def search_technical(query: str, n_results: int = 5) -> dict[str, Any]:
    """Search LAMMPS example input scripts for concrete syntax.

    Best for: finding working examples of a particular simulation type
    ("Lennard-Jones melt", "water TIP4P", "crack propagation", "GCMC
    insertion"), seeing how commands are combined in practice, and
    getting a structural template to adapt.

    Results point to example scripts under /lammps_lib/examples/. Always
    read the full script with Read before adapting it.
    """
    n_results = _bounded_n_results(n_results, default=5)
    collection = _get_backend().get_collection(COLLECTION_TECHNICAL)
    results = collection.query(query_texts=[query], n_results=n_results)
    policy = _get_policy()

    formatted: list[dict[str, Any]] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    for index, doc in enumerate(documents):
        meta = metadatas[index]
        in_ref = meta.get("in_reference") or ""
        source_path = meta.get("source_path", "")
        if policy.is_blocked_in_path(in_ref):
            continue
        if policy.is_blocked_in_path(source_path):
            continue
        if policy.is_blocked_rst_path(source_path):
            continue
        formatted.append(
            {
                "title": meta.get("title", "No Title"),
                "in_reference": in_ref,
                "line_range": meta.get("line_range", ""),
                "breadcrumbs": meta.get("breadcrumbs", ""),
                "source_path": source_path,
                "shadow_text": doc[:300] + "..." if len(doc) > 300 else doc,
            }
        )

    return {
        "query": query,
        "results": formatted,
        "hint": (
            "Use Read with the source path to read the actual input script. "
            "Adapt the example to the task spec rather than copying verbatim."
        ),
    }


@mcp.tool()
def search_commands(query: str, n_results: int = 3) -> dict[str, Any]:
    """Search LAMMPS command reference for exact syntax and keyword names.

    Best for: checking the exact arguments for a command ("fix nvt keywords",
    "pair_style lj/cut arguments", "compute msd syntax"), resolving ambiguity
    between similar commands, and finding required vs. optional keywords.

    Results come from the LAMMPS doc/src/ RST command pages.
    """
    n_results = _bounded_n_results(n_results, default=3)
    collection = _get_backend().get_collection(COLLECTION_COMMANDS)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    formatted: list[dict[str, Any]] = []
    for doc, meta, dist in zip(
        results.get("documents", [[]])[0],
        results.get("metadatas", [[]])[0],
        results.get("distances", [[]])[0],
    ):
        formatted.append(
            {
                "command": meta.get("command_name", meta.get("title", "unknown")),
                "title": meta.get("title", ""),
                "syntax_preview": doc[:400] + "..." if len(doc) > 400 else doc,
                "source": meta.get("source_path", ""),
                "relevance": round(1 - dist, 4),
            }
        )

    return {
        "query": query,
        "results": formatted,
        "hint": (
            "The syntax_preview field contains the command signature and key options. "
            "Read the full doc page for complete keyword tables and examples."
        ),
    }


def _smoke() -> int:
    vector_db_dir = _vector_db_dir()
    client = chromadb.PersistentClient(path=str(vector_db_dir))
    names = sorted(collection.name for collection in client.list_collections())
    print(f"vector_db_dir={vector_db_dir}")
    print("collections=" + ",".join(names))
    missing = {
        COLLECTION_NAVIGATOR,
        COLLECTION_TECHNICAL,
        COLLECTION_COMMANDS,
    } - set(names)
    if missing:
        print("missing=" + ",".join(sorted(missing)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    if "--smoke" in sys.argv:
        raise SystemExit(_smoke())
    mcp.run()

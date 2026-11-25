r"""Embed processed legal chunks and store them in Qdrant.

Default input:
    D:\Projects\RAG\data\processed\chunks\labor_law_legal.jsonl

Examples
--------
DEk21:
    python scripts/index_labor_law_chunks.py --model dek21

BGE-M3 dense:
    python scripts/index_labor_law_chunks.py --model bge

Use a different Qdrant URL:
    python scripts/index_labor_law_chunks.py ^
        --model dek21 ^
        --qdrant-url http://localhost:6333

Rebuild the collection from scratch:
    python scripts/index_labor_law_chunks.py --model dek21 --recreate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Allow direct execution from scripts/ while importing src/labor_rag.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_DATA_PATH = Path(
    r"D:\Projects\RAG\data\processed\chunks\labor_law_legal.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed legal chunks and upsert vector + payload into Qdrant."
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the processed chunk JSONL file.",
    )

    parser.add_argument(
        "--model",
        choices=["dek21", "bge"],
        default="dek21",
        help="Dense embedding model used for this collection.",
    )

    parser.add_argument(
        "--collection",
        default=None,
        help=(
            "Qdrant collection name. If omitted, a model-specific default "
            "name is used."
        ),
    )

    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant server URL.",
    )

    parser.add_argument(
        "--qdrant-path",
        default=None,
        help="Embedded local Qdrant directory (no server, e.g. on Kaggle). Overrides --qdrant-url.",
    )

    parser.add_argument(
        "--qdrant-api-key",
        default=None,
        help="Optional Qdrant API key.",
    )

    parser.add_argument(
        "--device",
        default=None,
        help='Embedding device, e.g. "cpu", "cuda", or "cuda:0".',
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size.",
    )

    parser.add_argument(
        "--dek21-dimension",
        type=int,
        choices=[768, 512, 256],
        default=768,
        help="DEk21 output dimension.",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the Qdrant collection before indexing.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for smoke testing, e.g. --limit 10.",
    )

    return parser.parse_args()


def load_chunks(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load processed chunks from a JSONL file."""
    path = path.expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    chunks: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {exc}"
                ) from exc

            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Line {line_number} must contain a JSON object."
                )

            validate_chunk(chunk, line_number)
            chunks.append(chunk)

            if limit is not None and len(chunks) >= limit:
                break

    if not chunks:
        raise ValueError(f"No chunks were loaded from {path}")

    return chunks


def validate_chunk(chunk: dict[str, Any], line_number: int) -> None:
    """Validate fields required by the current indexing pipeline."""
    chunk_id = chunk.get("chunk_id")
    embedding_text = chunk.get("embedding_text")

    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise ValueError(
            f"Line {line_number}: missing/non-string 'chunk_id'."
        )

    if not isinstance(embedding_text, str) or not embedding_text.strip():
        raise ValueError(
            f"Line {line_number}: chunk {chunk_id!r} has no "
            "'embedding_text'."
        )


def build_embedder(args: argparse.Namespace):
    """Create the selected dense embedder."""
    if args.model == "dek21":
        from labor_rag.embeddings.dense.dek21 import DEk21V2Embedder

        return DEk21V2Embedder(
            dimension=args.dek21_dimension,
            device=args.device,
            batch_size=args.batch_size,
            normalize_embeddings=True,
        )

    from labor_rag.embeddings.dense.bge import BGEM3DenseEmbedder

    return BGEM3DenseEmbedder(
        device=args.device,
        batch_size=args.batch_size,
        normalize_embeddings=True,
    )


def default_collection_name(args: argparse.Namespace) -> str:
    if args.model == "dek21":
        return f"labor_law_dek21_{args.dek21_dimension}"

    return "labor_law_bge_m3_dense"


def print_chunk_summary(chunks: list[dict[str, Any]]) -> None:
    """Print a small summary before running the expensive embedding step."""
    print(f"Loaded chunks: {len(chunks)}")

    levels: dict[str, int] = {}

    for chunk in chunks:
        level = str(chunk.get("chunk_level") or "unknown")
        levels[level] = levels.get(level, 0) + 1

    print("Chunk levels:")
    for level, count in sorted(levels.items()):
        print(f"  {level}: {count}")

    first = chunks[0]

    print("\nFirst chunk:")
    print(f"  chunk_id       : {first.get('chunk_id')}")
    print(f"  chunk_level    : {first.get('chunk_level')}")
    print(f"  article_number : {first.get('article_number')}")
    print(f"  clause_number  : {first.get('clause_number')}")
    print(
        "  embedding_text: "
        + str(first.get("embedding_text", ""))[:150].replace("\n", " ")
        + "..."
    )


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than 0.")

    data_path = args.data.expanduser()
    collection_name = args.collection or default_collection_name(args)

    print("=" * 72)
    print("LABOR LAW -> EMBEDDING -> QDRANT")
    print("=" * 72)
    print(f"Input file : {data_path}")
    print(f"Model      : {args.model}")
    print(f"Collection : {collection_name}")
    print(f"Qdrant URL : {args.qdrant_url}")
    print(f"Recreate   : {args.recreate}")
    print()

    # 1. Load already-processed chunks.
    chunks = load_chunks(
        data_path,
        limit=args.limit,
    )
    print_chunk_summary(chunks)

    # 2. Load embedding model.
    print("\nLoading embedding model...")
    embedder = build_embedder(args)

    print(f"Embedding model: {embedder.model_name}")
    print(f"Vector dimension: {embedder.dimension}")

    # 3. Connect to Qdrant.
    from labor_rag.storage.qdrant import QdrantStore

    store = QdrantStore(
        collection_name=collection_name,
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
        path=args.qdrant_path,
    )

    # 4. Run the project's simple indexing flow:
    #       chunk.embedding_text -> vector
    #       entire chunk          -> payload
    #       vector + payload      -> Qdrant
    from labor_rag.indexing.indexer import index_chunks

    print("\nEmbedding chunks and writing them to Qdrant...")

    indexed_count = index_chunks(
        chunks=chunks,
        embedder=embedder,
        store=store,
        recreate_collection=args.recreate,
    )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Indexed chunks : {indexed_count}")
    print(f"Collection     : {collection_name}")
    print(f"Vector size    : {embedder.dimension}")
    print(
        "\nFor every Qdrant point:\n"
        "  vector  = embedding of chunk['embedding_text']\n"
        "  payload = the full original chunk JSON object"
    )


if __name__ == "__main__":
    main()

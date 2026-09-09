r"""Run dense, BM25, or hybrid retrieval for the labor-law corpus.

Queries are stored in:
    configs/retrieval_queries.json

PowerShell examples
-------------------
List saved queries:
    python scripts/retrieve_labor_law.py --list-queries

Dense retrieval with a saved query:
    python scripts/retrieve_labor_law.py --retrieval dense --query-id annual_leave

BM25 retrieval:
    python scripts/retrieve_labor_law.py --retrieval bm25 --query-id annual_leave

Hybrid dense + BM25 retrieval:
    python scripts/retrieve_labor_law.py --retrieval hybrid --query-id annual_leave

Hybrid retrieval and save candidates for reranking:
    python scripts/retrieve_labor_law.py `
        --retrieval hybrid `
        --query-id annual_leave `
        --top-k 20 `
        --save-candidates

One-off query without editing the query file:
    python scripts/retrieve_labor_law.py --retrieval hybrid `
        --query "Người lao động được nghỉ hằng năm bao nhiêu ngày?" `
        --top-k 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "labor_law_legal.jsonl"
)
DEFAULT_QUERIES_PATH = PROJECT_ROOT / "configs" / "retrieval_queries.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "retrieval_results"

DEFAULT_QUERY_ID = "annual_leave"

SUPPORTED_MODELS = ("bge", "dek21")
SUPPORTED_RETRIEVALS = ("dense", "bm25", "hybrid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve labor-law chunks with dense, BM25, or hybrid search."
    )

    parser.add_argument(
        "--retrieval",
        choices=SUPPORTED_RETRIEVALS,
        default="hybrid",
        help="Retrieval strategy. Default: hybrid.",
    )
    parser.add_argument(
        "--query-id",
        default=DEFAULT_QUERY_ID,
        help=(
            "Query key from configs/retrieval_queries.json. "
            f"Default: {DEFAULT_QUERY_ID}."
        ),
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional one-off query. When provided, it overrides --query-id.",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help="JSON file containing reusable named queries.",
    )
    parser.add_argument(
        "--list-queries",
        action="store_true",
        help="Print all saved query IDs and exit.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of final chunks to return.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
        help="Candidates taken from each branch before hybrid RRF fusion.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Processed chunk JSONL used to build the BM25 index.",
    )
    parser.add_argument(
        "--bm25-k1",
        type=float,
        default=1.5,
        help="BM25 term-frequency saturation parameter. Default: 1.5.",
    )
    parser.add_argument(
        "--bm25-b",
        type=float,
        default=0.75,
        help="BM25 document-length normalization parameter. Default: 0.75.",
    )

    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="bge",
        help="Dense embedding model. Used by dense/hybrid retrieval.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection. If omitted, inferred from --model.",
    )
    parser.add_argument(
        "--qdrant-url",
        default="http://localhost:6333",
        help="Qdrant server URL.",
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
        "--dek21-dimension",
        type=int,
        choices=[768, 512, 256],
        default=768,
        help="DEk21 dimension. Must match the indexed collection.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF rank constant. Default: 60.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=1.0,
        help="Dense branch weight in weighted RRF. Default: 1.0.",
    )
    parser.add_argument(
        "--bm25-weight",
        type=float,
        default=1.0,
        help="BM25 branch weight in weighted RRF. Default: 1.0.",
    )

    parser.add_argument(
        "--save-candidates",
        action="store_true",
        help=(
            "Save retrieval results as JSON so rerank_labor_law.py can "
            "reuse them without running retrieval again."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Candidate JSON output path. Supplying --output also enables saving. "
            "If omitted with --save-candidates, a filename is generated automatically."
        ),
    )

    return parser.parse_args()


def resolve_project_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_queries(path: Path) -> dict[str, str]:
    path = resolve_project_path(path)

    if not path.exists():
        raise FileNotFoundError(f"Queries file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    if not isinstance(raw, dict):
        raise ValueError("Queries file must contain a JSON object.")

    queries: dict[str, str] = {}
    for query_id, query in raw.items():
        if (
            isinstance(query_id, str)
            and isinstance(query, str)
            and query.strip()
        ):
            queries[query_id] = query.strip()

    if not queries:
        raise ValueError(f"No valid queries found in {path}")

    return queries


def resolve_query(
    args: argparse.Namespace,
    queries: dict[str, str],
) -> tuple[str, str]:
    if args.query is not None:
        query = args.query.strip()
        if not query:
            raise ValueError("--query must not be empty.")
        return "custom", query

    if args.query_id not in queries:
        available = ", ".join(sorted(queries))
        raise ValueError(
            f"Unknown --query-id {args.query_id!r}. Available: {available}"
        )

    return args.query_id, queries[args.query_id]


def build_embedder(args: argparse.Namespace):
    if args.model == "dek21":
        from labor_rag.embeddings.dense.dek21 import DEk21V2Embedder

        return DEk21V2Embedder(
            dimension=args.dek21_dimension,
            device=args.device,
            normalize_embeddings=True,
        )

    from labor_rag.embeddings.dense.bge import BGEM3DenseEmbedder

    return BGEM3DenseEmbedder(
        device=args.device,
        normalize_embeddings=True,
    )


def default_collection_name(args: argparse.Namespace) -> str:
    if args.model == "dek21":
        return f"labor_law_dek21_{args.dek21_dimension}"

    return "labor_law_bge_m3_dense"


def build_dense_retriever(args: argparse.Namespace):
    from labor_rag.retrieval.dense_retriever import DenseRetriever
    from labor_rag.storage.qdrant import QdrantStore

    collection_name = args.collection or default_collection_name(args)
    embedder = build_embedder(args)

    store = QdrantStore(
        collection_name=collection_name,
        url=args.qdrant_url,
        api_key=args.qdrant_api_key,
    )

    return (
        DenseRetriever(
            embedder=embedder,
            store=store,
        ),
        embedder,
        collection_name,
    )


def build_bm25_retriever(args: argparse.Namespace):
    from labor_rag.retrieval.bm25_retriever import BM25Retriever

    return BM25Retriever.from_jsonl(
        resolve_project_path(args.data),
        k1=args.bm25_k1,
        b=args.bm25_b,
    )


def build_retriever(args: argparse.Namespace):
    if args.retrieval == "bm25":
        return build_bm25_retriever(args), None, None

    dense_retriever, embedder, collection_name = build_dense_retriever(args)

    if args.retrieval == "dense":
        return dense_retriever, embedder, collection_name

    from labor_rag.retrieval.hybrid_retriever import HybridRetriever

    bm25_retriever = build_bm25_retriever(args)

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        candidate_k=args.candidate_k,
        rrf_k=args.rrf_k,
        dense_weight=args.dense_weight,
        bm25_weight=args.bm25_weight,
    )

    return hybrid_retriever, embedder, collection_name


def print_result(rank: int, result) -> None:
    meta = result.metadata

    article = meta.get("article_number")
    clause = meta.get("clause_number")
    title = meta.get("article_title")

    legal_ref = f"Điều {article}" if article else "Không rõ điều"
    if clause:
        legal_ref += f", khoản {clause}"

    print(f"\n[{rank}] score={result.score:.6f} | {legal_ref}")

    if title:
        print(f"    {title}")

    print(f"    chunk_id: {result.chunk_id}")

    if meta.get("retrieval_method") == "hybrid_rrf":
        dense_rank = meta.get("dense_rank")
        bm25_rank = meta.get("bm25_rank")
        dense_score = meta.get("dense_score")
        bm25_score = meta.get("bm25_score")

        print(
            "    hybrid: "
            f"dense_rank={dense_rank}, dense_score={dense_score}; "
            f"bm25_rank={bm25_rank}, bm25_score={bm25_score}"
        )

    print("-" * 80)
    print(
        result.content.strip()
        or str(meta.get("embedding_text", "")).strip()
    )


def safe_filename(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "query"


def default_output_path(
    args: argparse.Namespace,
    query_id: str,
) -> Path:
    name_parts = [
        safe_filename(query_id),
        args.retrieval,
    ]

    if args.retrieval in {"dense", "hybrid"}:
        name_parts.append(args.model)

        if args.model == "dek21":
            name_parts.append(str(args.dek21_dimension))

    filename = "__".join(name_parts) + ".json"
    return DEFAULT_OUTPUT_DIR / filename


def candidate_to_dict(rank: int, result) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": result.chunk_id,
        "content": result.content,
        "score": float(result.score),
        "metadata": dict(result.metadata),
    }


def build_save_payload(
    *,
    args: argparse.Namespace,
    query_id: str,
    query: str,
    results,
    embedder,
    collection_name: str | None,
) -> dict[str, Any]:
    retrieval_config: dict[str, Any] = {
        "method": args.retrieval,
        "top_k": args.top_k,
    }

    if args.retrieval in {"dense", "hybrid"}:
        retrieval_config.update(
            {
                "model": args.model,
                "collection": collection_name,
                "device": args.device,
            }
        )

        if embedder is not None:
            retrieval_config["model_name"] = embedder.model_name
            retrieval_config["dimension"] = embedder.dimension

        if args.model == "dek21":
            retrieval_config["dek21_dimension"] = args.dek21_dimension

    if args.retrieval in {"bm25", "hybrid"}:
        retrieval_config.update(
            {
                "bm25_data": str(resolve_project_path(args.data)),
                "bm25_k1": args.bm25_k1,
                "bm25_b": args.bm25_b,
            }
        )

    if args.retrieval == "hybrid":
        retrieval_config.update(
            {
                "candidate_k": args.candidate_k,
                "rrf_k": args.rrf_k,
                "dense_weight": args.dense_weight,
                "bm25_weight": args.bm25_weight,
            }
        )

    return {
        "query_id": query_id,
        "query": query,
        "retrieval": retrieval_config,
        "candidate_count": len(results),
        "candidates": [
            candidate_to_dict(rank, result)
            for rank, result in enumerate(results, start=1)
        ],
    }


def save_candidates(
    *,
    args: argparse.Namespace,
    query_id: str,
    query: str,
    results,
    embedder,
    collection_name: str | None,
) -> Path:
    output_path = (
        resolve_project_path(args.output)
        if args.output is not None
        else default_output_path(args, query_id)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_save_payload(
        args=args,
        query_id=query_id,
        query=query,
        results=results,
        embedder=embedder,
        collection_name=collection_name,
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    return output_path


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")

    if args.candidate_k <= 0:
        raise ValueError("--candidate-k must be greater than 0.")

    if args.bm25_k1 <= 0:
        raise ValueError("--bm25-k1 must be greater than 0.")

    if not 0 <= args.bm25_b <= 1:
        raise ValueError("--bm25-b must be between 0 and 1.")

    if args.rrf_k < 0:
        raise ValueError("--rrf-k must be non-negative.")

    if args.dense_weight < 0 or args.bm25_weight < 0:
        raise ValueError(
            "--dense-weight and --bm25-weight must be non-negative."
        )

    if (
        args.retrieval == "hybrid"
        and args.dense_weight == 0
        and args.bm25_weight == 0
    ):
        raise ValueError(
            "At least one of --dense-weight or --bm25-weight must be positive."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    queries = load_queries(args.queries_file)

    if args.list_queries:
        print(
            f"Queries file: "
            f"{resolve_project_path(args.queries_file)}"
        )

        for query_id, query in queries.items():
            print(f"  {query_id}: {query}")

        return

    query_id, query = resolve_query(args, queries)

    retriever, embedder, collection_name = build_retriever(args)

    results = retriever.retrieve(
        query,
        top_k=args.top_k,
    )

    print("=" * 80)
    print(f"QUERY ID   : {query_id}")
    print(f"QUERY      : {query}")
    print(f"RETRIEVAL  : {args.retrieval}")
    print(f"TOP K      : {args.top_k}")

    if embedder is not None:
        print(f"MODEL      : {embedder.model_name}")
        print(f"DIMENSION  : {embedder.dimension}")
        print(f"COLLECTION : {collection_name}")

    if args.retrieval in {"bm25", "hybrid"}:
        print(f"BM25 DATA  : {resolve_project_path(args.data)}")
        print(f"BM25 K1/B  : {args.bm25_k1} / {args.bm25_b}")

    if args.retrieval == "hybrid":
        print(f"CANDIDATE K: {args.candidate_k} per retriever")
        print(f"RRF K      : {args.rrf_k}")
        print(
            "RRF WEIGHTS: "
            f"dense={args.dense_weight}, "
            f"bm25={args.bm25_weight}"
        )

    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        print_result(rank, result)

    should_save = args.save_candidates or args.output is not None

    if should_save:
        output_path = save_candidates(
            args=args,
            query_id=query_id,
            query=query,
            results=results,
            embedder=embedder,
            collection_name=collection_name,
        )

        print("\n" + "=" * 80)
        print(f"SAVED      : {output_path}")
        print(f"CANDIDATES : {len(results)}")
        print("=" * 80)


if __name__ == "__main__":
    main()

r"""Rerank saved labor-law retrieval candidates with BGE reranker.

Typical flow
------------
1. Retrieve and save candidates:
    python scripts/retrieve_labor_law.py `
        --retrieval hybrid `
        --query-id annual_leave `
        --top-k 20 `
        --save-candidates

2. Rerank and save results for Context Builder:
    python scripts/rerank_labor_law.py `
        --input data/retrieval_results/annual_leave__hybrid__bge.json `
        --top-k 10 `
        --save-results

3. Build context:
    python scripts/build_labor_law_context.py `
        --input data/reranking_results/annual_leave__hybrid__bge__reranked.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_INPUT = PROJECT_ROOT / "data" / "retrieval_results" / "candidates.json"
DEFAULT_QUERIES_FILE = PROJECT_ROOT / "configs" / "retrieval_queries.json"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "reranking_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank saved labor-law retrieval candidates with BGE."
    )

    query_group = parser.add_mutually_exclusive_group(required=False)
    query_group.add_argument(
        "--query",
        default=None,
        help="Query used to rerank the saved candidates.",
    )
    query_group.add_argument(
        "--query-id",
        default=None,
        help="Query key from configs/retrieval_queries.json.",
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="JSON file containing saved retrieval candidates.",
    )
    parser.add_argument(
        "--queries-file",
        default=str(DEFAULT_QUERIES_FILE),
        help="JSON file mapping query IDs to query text.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help=(
            "Number of reranked candidates to keep. Keep more than the final "
            "generation context so Context Builder can cut adaptively. Default: 10."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_RERANKER_MODEL,
        help="Cross-encoder reranker model.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Reranker device, e.g. "cpu", "cuda", or "cuda:0".',
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Reranker inference batch size.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length for each query-document pair.",
    )
    parser.add_argument(
        "--raw-scores",
        action="store_true",
        help=(
            "Keep raw reranker logits instead of sigmoid scores. Do not use this "
            "when the next step is threshold-based Context Builder."
        ),
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save reranked results as JSON for build_labor_law_context.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Reranked JSON output path. Supplying --output also enables saving. "
            "Default: data/reranking_results/<input>__reranked.json."
        ),
    )

    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_query_from_id(query_id: str, queries_file: Path) -> str:
    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    with queries_file.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    if not isinstance(queries, dict):
        raise ValueError(
            f"Queries file must contain a JSON object: {queries_file}"
        )

    query = queries.get(query_id)
    if not isinstance(query, str) or not query.strip():
        available = ", ".join(sorted(str(key) for key in queries))
        raise KeyError(
            f"Unknown query-id {query_id!r}. Available query IDs: {available}"
        )

    return query.strip()


def load_candidate_payload(
    path: Path,
) -> tuple[str | None, str | None, list[dict[str, Any]], dict[str, Any] | None]:
    """Load either a plain candidate list or a retrieval envelope object."""
    if not path.exists():
        raise FileNotFoundError(
            f"Candidates file not found: {path}\n"
            "Run retrieve_labor_law.py with --save-candidates first."
        )

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    saved_query: str | None = None
    saved_query_id: str | None = None
    source_retrieval: dict[str, Any] | None = None

    if isinstance(payload, list):
        raw_candidates = payload
    elif isinstance(payload, dict):
        query_value = payload.get("query")
        if isinstance(query_value, str) and query_value.strip():
            saved_query = query_value.strip()

        query_id_value = payload.get("query_id")
        if isinstance(query_id_value, str) and query_id_value.strip():
            saved_query_id = query_id_value.strip()

        retrieval_value = payload.get("retrieval")
        if isinstance(retrieval_value, dict):
            source_retrieval = dict(retrieval_value)

        raw_candidates = payload.get("candidates")
        if raw_candidates is None:
            raw_candidates = payload.get("results")

        if not isinstance(raw_candidates, list):
            raise ValueError(
                "Candidate JSON object must contain a 'candidates' "
                "or 'results' list."
            )
    else:
        raise ValueError(
            "Candidates JSON must be either a list or an object "
            "containing a candidates/results list."
        )

    for index, item in enumerate(raw_candidates):
        if not isinstance(item, dict):
            raise ValueError(
                f"Candidate at index {index} must be a JSON object."
            )

    return saved_query, saved_query_id, raw_candidates, source_retrieval


def build_candidates(raw_candidates: list[dict[str, Any]]):
    from labor_rag.retrieval.base import RetrievalResult

    candidates: list[RetrievalResult] = []

    for index, item in enumerate(raw_candidates):
        content = str(
            item.get("content")
            or (item.get("metadata") or {}).get("embedding_text")
            or ""
        ).strip()

        if not content:
            raise ValueError(f"Candidate at index {index} has no content.")

        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Candidate metadata at index {index} must be an object."
            )

        candidates.append(
            RetrievalResult(
                chunk_id=str(item.get("chunk_id") or index),
                content=content,
                score=float(item.get("score", 0.0)),
                metadata=dict(metadata),
            )
        )

    return candidates


def resolve_query(
    args: argparse.Namespace,
    saved_query: str | None,
) -> str:
    if args.query:
        return args.query.strip()

    if args.query_id:
        return load_query_from_id(
            args.query_id,
            resolve_path(args.queries_file),
        )

    if saved_query:
        return saved_query

    raise ValueError(
        "No query was provided. Use --query or --query-id, "
        "or save the query inside candidates.json."
    )


def print_results(query: str, results) -> None:
    print("=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    if not results:
        print("No candidates found.")
        return

    for rank, result in enumerate(results, start=1):
        meta = result.metadata

        article = meta.get("article_number")
        clause = meta.get("clause_number")
        title = meta.get("article_title")

        legal_ref = f"Điều {article}" if article else "Không rõ điều"
        if clause:
            legal_ref += f", khoản {clause}"

        retrieval_score = meta.get("retrieval_score")

        print(
            f"\n[{rank}] rerank_score={result.score:.6f}"
            f" | {legal_ref}"
        )

        if retrieval_score is not None:
            print(f"    retrieval_score: {float(retrieval_score):.6f}")

        if title:
            print(f"    {title}")

        print(f"    chunk_id: {result.chunk_id}")
        print("-" * 80)
        print(
            result.content.strip()
            or str(meta.get("embedding_text", "")).strip()
        )


def result_to_dict(rank: int, result) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": result.chunk_id,
        "content": result.content,
        "score": float(result.score),
        "reranker_score": float(result.score),
        "metadata": dict(result.metadata),
    }


def default_output_path(input_path: Path) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{input_path.stem}__reranked.json"


def save_results(
    *,
    args: argparse.Namespace,
    input_path: Path,
    query: str,
    query_id: str | None,
    source_retrieval: dict[str, Any] | None,
    results,
) -> Path:
    output_path = (
        resolve_path(args.output)
        if args.output is not None
        else default_output_path(input_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "query_id": query_id,
        "query": query,
        "source_candidate_file": str(input_path),
        "source_retrieval": source_retrieval,
        "reranking": {
            "model": args.model_name,
            "top_k": args.top_k,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "device": args.device,
            "normalized_scores": not args.raw_scores,
            "score_transform": "sigmoid" if not args.raw_scores else "none",
        },
        "result_count": len(results),
        "results": [
            result_to_dict(rank, result)
            for rank, result in enumerate(results, start=1)
        ],
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    return output_path


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than 0.")
    if args.max_length <= 0:
        raise ValueError("--max-length must be greater than 0.")

    input_path = resolve_path(args.input)
    saved_query, saved_query_id, raw_candidates, source_retrieval = (
        load_candidate_payload(input_path)
    )

    query = resolve_query(args, saved_query)
    query_id = args.query_id or saved_query_id
    candidates = build_candidates(raw_candidates)

    from labor_rag.reranking.bge_reranker import BGEReranker

    reranker = BGEReranker(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        normalize_scores=not args.raw_scores,
    )

    results = reranker.rerank(
        query,
        candidates,
        top_k=args.top_k,
    )

    print_results(query, results)

    if args.save_results or args.output is not None:
        output_path = save_results(
            args=args,
            input_path=input_path,
            query=query,
            query_id=query_id,
            source_retrieval=source_retrieval,
            results=results,
        )
        print("\n" + "=" * 80)
        print(f"SAVED      : {output_path}")
        print(f"RESULTS    : {len(results)}")
        print("=" * 80)


if __name__ == "__main__":
    main()

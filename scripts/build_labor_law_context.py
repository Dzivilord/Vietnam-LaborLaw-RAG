r"""Build generation context from a saved reranked JSON file.

Expected pipeline
-----------------
1. Retrieval -> save candidates JSON.
2. Reranking -> save reranked JSON.
3. This script -> select evidence and save context JSON.

PowerShell example
------------------
python scripts/build_labor_law_context.py `
    --input data/reranking_results/annual_leave__reranked.json `
    --min-score 0.60 `
    --score-gap 0.20 `
    --max-chunks 5 `
    --max-tokens 3500
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


DEFAULT_INPUT = PROJECT_ROOT / "data" / "reranking_results" / "reranked.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "context_results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LLM-ready context from saved BGE reranker results."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Reranked JSON produced by rerank_labor_law.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output context JSON. Default: data/context_results/<input>__context.json.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.60,
        help="Absolute normalized reranker-score threshold. Default: 0.60.",
    )
    parser.add_argument(
        "--relative-to-top",
        type=float,
        default=None,
        help=(
            "Optional relative threshold: keep score >= top_score * value. "
            "Disabled by default."
        ),
    )
    parser.add_argument(
        "--score-gap",
        type=float,
        default=0.20,
        help=(
            "Adaptive cutoff when consecutive reranker scores drop by at least "
            "this amount. Use a negative value to disable. Default: 0.20."
        ),
    )
    parser.add_argument(
        "--min-chunks-before-gap",
        type=int,
        default=1,
        help="Do not apply a score-gap cutoff before this many chunks are kept.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=5,
        help="Hard maximum number of context chunks. Default: 5.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=3500,
        help=(
            "Approximate context token budget. Uses metadata token_count when "
            "available. Use 0 to disable. Default: 3500."
        ),
    )
    parser.add_argument(
        "--order-by",
        choices=("legal", "relevance"),
        default="legal",
        help="Final context ordering. Default: legal.",
    )
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_reranked_payload(path: Path) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Reranked JSON not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("Reranked JSON must be an object envelope.")

    query = payload.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Reranked JSON must contain a non-empty 'query'.")

    reranking = payload.get("reranking") or {}
    if not isinstance(reranking, dict):
        raise ValueError("'reranking' must be a JSON object when present.")

    normalized = reranking.get("normalized_scores")
    if normalized is False:
        raise ValueError(
            "Context score thresholds require normalized reranker scores. "
            "Rerun rerank_labor_law.py without --raw-scores."
        )

    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise ValueError("Reranked JSON must contain a 'results' list.")

    for index, item in enumerate(raw_results):
        if not isinstance(item, dict):
            raise ValueError(f"results[{index}] must be a JSON object.")

    return query.strip(), raw_results, payload


def build_results(raw_results: list[dict[str, Any]]):
    from labor_rag.retrieval.base import RetrievalResult

    results: list[RetrievalResult] = []
    for index, item in enumerate(raw_results):
        metadata = item.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"results[{index}].metadata must be an object.")

        content = str(
            item.get("content")
            or metadata.get("embedding_text")
            or ""
        ).strip()
        if not content:
            raise ValueError(f"results[{index}] has no content.")

        score_value = item.get("reranker_score", item.get("score"))
        if score_value is None:
            score_value = metadata.get("reranker_score")
        if score_value is None:
            raise ValueError(f"results[{index}] has no reranker score.")

        results.append(
            RetrievalResult(
                chunk_id=str(item.get("chunk_id") or index),
                content=content,
                score=float(score_value),
                metadata=dict(metadata),
            )
        )

    return results


def selection_to_dict(selection) -> dict[str, Any]:
    result = selection.result
    return {
        "rerank_rank": selection.rerank_rank,
        "chunk_id": result.chunk_id,
        "reranker_score": float(result.score),
        "estimated_tokens": selection.estimated_tokens,
        "content": result.content,
        "metadata": dict(result.metadata),
    }


def default_output_path(input_path: Path) -> Path:
    name = input_path.stem
    if name.endswith("__reranked"):
        name = name[: -len("__reranked")]
    return DEFAULT_OUTPUT_DIR / f"{name}__context.json"


def main() -> None:
    args = parse_args()

    score_gap = None if args.score_gap < 0 else args.score_gap
    max_tokens = None if args.max_tokens == 0 else args.max_tokens

    input_path = resolve_path(args.input)
    query, raw_results, source_payload = load_reranked_payload(input_path)
    results = build_results(raw_results)

    from labor_rag.reranking.context_builder import (
        ContextBuilderConfig,
        RerankedContextBuilder,
    )

    config = ContextBuilderConfig(
        min_score=args.min_score,
        min_relative_to_top=args.relative_to_top,
        score_gap=score_gap,
        min_chunks_before_gap=args.min_chunks_before_gap,
        max_chunks=args.max_chunks,
        max_tokens=max_tokens,
        order_by=args.order_by,
    )
    build_result = RerankedContextBuilder(config).build(query, results)

    output_path = (
        resolve_path(args.output)
        if args.output is not None
        else default_output_path(input_path)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "query_id": source_payload.get("query_id"),
        "query": query,
        "status": build_result.status,
        "source_reranked_file": str(input_path),
        "selection": build_result.diagnostics,
        "selected_chunks": [
            selection_to_dict(item) for item in build_result.selected
        ],
        "rejected_chunks": build_result.rejected,
        "context": build_result.context,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"QUERY       : {query}")
    print(f"STATUS      : {build_result.status}")
    print(f"INPUT       : {len(results)} reranked chunks")
    print(f"SELECTED    : {len(build_result.selected)} chunks")
    print(f"REJECTED    : {len(build_result.rejected)} chunks")
    print(f"OUTPUT      : {output_path}")
    print("=" * 80)

    if build_result.status == "insufficient_context":
        print("No sufficiently relevant context. Generation should abstain/clarify.")
        return

    for item in build_result.selected:
        meta = item.result.metadata
        article = meta.get("article_number")
        clause = meta.get("clause_number")
        legal_ref = f"Điều {article}" if article else item.result.chunk_id
        if clause:
            legal_ref += f", khoản {clause}"
        print(
            f"rank={item.rerank_rank} score={item.result.score:.6f} "
            f"| {legal_ref}"
        )

    print("\nCONTEXT")
    print("-" * 80)
    print(build_result.context)


if __name__ == "__main__":
    main()

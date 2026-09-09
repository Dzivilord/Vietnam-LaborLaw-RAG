r"""Interactive retrieval -> rerank -> context pipeline for Vietnamese labor law.

This runner now keeps data in memory.  JSONL is optional observability output
only; generation no longer depends on it.

Examples
--------
    python scripts/run_rag_pipeline.py
    python scripts/run_rag_pipeline.py --reranker qwen3 --check-ambiguity
    python scripts/run_rag_pipeline.py --no-rerank --no-rerank-context-k 5
    python scripts/run_rag_pipeline.py --retrieval dense --embedding-model dek21
    python scripts/run_rag_pipeline.py --query "thử việc"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from labor_rag.models import json_safe
from labor_rag.runtime import (
    build_retrieval_context_pipeline,
    resolved_reranker_max_length,
    resolved_reranker_model,
)
from labor_rag.runtime.cli import add_core_pipeline_arguments, runtime_config_from_args


DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "chunks" / "labor_law_legal.jsonl"
EXIT_COMMANDS = {"exit", "quit", "q"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval -> rerank -> context directly in memory."
    )
    add_core_pipeline_arguments(parser, default_data_path=DEFAULT_DATA_PATH)
    parser.add_argument("--query", default=None, help="Run one query and exit.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help=(
            "Optional final-result JSONL for debugging/observability. It is not "
            "used as an intermediate input by generation."
        ),
    )
    return parser.parse_args()


def load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def append_log(path: Path | None, result) -> None:
    if path is None:
        return
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **json_safe(result),
    }
    with path.open("a", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, separators=(",", ":"))
        file.write("\n")


def legal_label(item) -> str:
    metadata = item.metadata
    article = metadata.get("article_number")
    clause = metadata.get("clause_number")
    title = str(metadata.get("article_title") or "").strip()
    reference = f"Điều {article}" if article else item.chunk_id
    if clause:
        reference += f", khoản {clause}"
    if title:
        reference += f" — {title}"
    return reference


def print_startup(config, embedder, collection: str | None, log_file: Path | None) -> None:
    print("=" * 92)
    print("LABOR LAW RAG — RETRIEVAL / CONTEXT")
    print(f"retrieval        : {config.retrieval.method}")
    print(f"retrieval top-k  : {config.retrieval.top_k}")
    if config.retrieval.method in {"dense", "hybrid"}:
        print(f"embedding model  : {getattr(embedder, 'model_name', config.retrieval.embedding_model)}")
        print(f"collection       : {collection}")
    if config.retrieval.method == "hybrid":
        print(f"candidate-k      : {config.retrieval.candidate_k}")
        print(f"rrf-k            : {config.retrieval.rrf_k}")
    print(f"rerank           : {'ON' if config.reranker.enabled else 'OFF'}")
    if config.reranker.enabled:
        print(f"reranker         : {config.reranker.type}")
        print(f"reranker model   : {resolved_reranker_model(config)}")
        print(f"rerank top-k     : {config.reranker.top_k}")
        print(f"reranker length  : {resolved_reranker_max_length(config)}")
        print(f"min score        : {config.context.min_score}")
    else:
        print(f"context fallback : top {config.reranker.no_rerank_context_k} retrieval results")
    print(f"context max      : {config.context.max_chunks} chunks / {config.context.max_tokens or 'unlimited'} tokens")
    print(f"ambiguity        : {'ON' if config.ambiguity.enabled else 'OFF'}")
    if config.ambiguity.enabled:
        print(f"ambiguity gate   : >= {config.ambiguity.min_relevant_chunks} context chunks")
    print(f"JSONL log        : {resolve_path(log_file) if log_file else 'OFF'}")
    print("=" * 92)


def print_result(result) -> None:
    print("\n" + "=" * 92)
    print(f"STATUS          : {result.status}")
    print(f"RETRIEVED       : {len(result.retrieval_results)}")
    print(f"RERANKED        : {len(result.reranked_results)}")
    print(f"CONTEXT CHUNKS  : {len(result.context_result.evidence)}")
    print(f"AMBIGUITY CHECK : {'TRIGGERED' if result.ambiguity_check_triggered else 'not triggered'}")
    print("=" * 92)

    if result.reranked_results:
        selected = {item.chunk_id for item in result.context_result.evidence}
        print("\nRERANK")
        for rank, item in enumerate(result.reranked_results, start=1):
            marker = "KEEP" if item.chunk_id in selected else "----"
            print(f"[{rank:>2}] {item.score:.6f}  {marker}  {legal_label(item)}")

    if result.status == "clarification_required":
        print("\nSUGGESTIONS")
        for index, suggestion in enumerate(result.suggestions, start=1):
            print(f"  {index}. {suggestion.label}")
        return

    if result.status == "insufficient_context":
        print("\nNo context survived selection.")
        return

    print("\nEVIDENCE")
    for evidence in result.context_result.evidence:
        ref = f"Điều {evidence.article_number}" if evidence.article_number else evidence.chunk_id
        if evidence.clause_number:
            ref += f", khoản {evidence.clause_number}"
        print(f"  [{evidence.evidence_id}] score={evidence.score:.6f} {ref}")

    print("\nCONTEXT")
    print("-" * 92)
    print(result.context)


def run_query(pipeline, query: str, log_file: Path | None) -> None:
    result = pipeline.run(query)
    print_result(result)
    append_log(log_file, result)


def main() -> None:
    args = parse_args()
    load_environment()
    config = runtime_config_from_args(
        args,
        project_root=PROJECT_ROOT,
        enable_generation=False,
    )
    pipeline, embedder, collection = build_retrieval_context_pipeline(config)
    print_startup(config, embedder, collection, args.log_file)

    if args.query:
        run_query(pipeline, args.query.strip(), args.log_file)
        return

    print("Enter a query. Type exit/quit/q to stop.\n")
    try:
        while True:
            try:
                query = input("Query> ").strip()
            except EOFError:
                print()
                break
            if not query:
                continue
            if query.casefold() in EXIT_COMMANDS:
                break
            try:
                run_query(pipeline, query, args.log_file)
            except Exception as exc:
                print(f"\nERROR: {type(exc).__name__}: {exc}\n", file=sys.stderr)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()

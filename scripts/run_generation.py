r"""Run the complete labor-law RAG flow in memory and generate an answer.

There is no JSONL hand-off anymore:
    query -> retrieval -> rerank -> ambiguity/context -> generation -> result

All stage options remain configurable.  JSONL output, if requested, is only a
final debug/observability log and is never read back by the pipeline.

Examples
--------
Ollama + Qwen 3.5 4B:
    python scripts/run_generation.py --model qwen3.5:4b

One-shot query:
    python scripts/run_generation.py --query "thử việc" --model qwen3.5:4b

Gemini:
    python scripts/run_generation.py --provider gemini --model <gemini-model>

Qwen3 reranker + ambiguity gate:
    python scripts/run_generation.py --reranker qwen3 --check-ambiguity --model qwen3.5:4b

DEk21 dense embedding while keeping a separate generation model:
    python scripts/run_generation.py --embedding-model dek21 --model qwen3.5:4b
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
from labor_rag.runtime import build_application_pipeline
from labor_rag.runtime.cli import (
    add_core_pipeline_arguments,
    add_generation_arguments,
    runtime_config_from_args,
)


DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "chunks" / "labor_law_legal.jsonl"
EXIT_COMMANDS = {"exit", "quit", "q"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full RAG + generation directly in memory."
    )
    # --model is kept as the generation-model alias in this script for backward
    # compatibility. Use --embedding-model for the retrieval embedding model.
    add_core_pipeline_arguments(
        parser,
        default_data_path=DEFAULT_DATA_PATH,
        include_legacy_model_alias=False,
    )
    add_generation_arguments(parser)
    parser.add_argument("--query", default=None, help="Run one query and exit.")
    parser.add_argument(
        "--output-log",
        type=Path,
        default=None,
        help="Optional final end-to-end JSONL log. Default: disabled.",
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


def print_startup(config, embedder, collection: str | None, output_log: Path | None) -> None:
    print("=" * 92)
    print("LABOR LAW RAG — END TO END")
    print(f"retrieval        : {config.retrieval.method}")
    print(f"embedding model  : {config.retrieval.embedding_model}")
    if collection:
        print(f"collection       : {collection}")
    print(f"retrieval top-k  : {config.retrieval.top_k}")
    print(f"rerank           : {'ON' if config.reranker.enabled else 'OFF'}")
    if config.reranker.enabled:
        print(f"reranker         : {config.reranker.type}")
        print(f"rerank top-k     : {config.reranker.top_k}")
    print(f"context max      : {config.context.max_chunks} chunks / {config.context.max_tokens or 'unlimited'} tokens")
    print(f"ambiguity        : {'ON' if config.ambiguity.enabled else 'OFF'}")
    print(f"provider         : {config.generation.provider}")
    print(f"generation model : {config.generation.model or '(provider default)'}")
    print(f"thinking         : {'ON' if config.generation.ollama_think else 'OFF'}")
    print(f"JSONL log        : {resolve_path(output_log) if output_log else 'OFF'}")
    print("=" * 92)


def print_result(result) -> None:
    print("\n" + "=" * 92)
    print(f"STATUS          : {result.status}")
    print(f"RETRIEVED       : {len(result.retrieval_results)}")
    print(f"RERANKED        : {len(result.reranked_results)}")
    print(f"EVIDENCE        : {len(result.evidence)}")

    if result.generation is not None:
        gen = result.generation
        print(f"PROVIDER        : {gen.provider}")
        print(f"MODEL           : {gen.model}")
        print(f"LATENCY         : {gen.latency_ms:.1f} ms")
        print(f"CITATIONS       : {', '.join(gen.citations_used) or '(none)'}")
        if gen.citation_warnings:
            print(f"WARNINGS        : {', '.join(gen.citation_warnings)}")
    print("=" * 92)

    if result.status == "clarification_required":
        print("\nSUGGESTIONS")
        for index, suggestion in enumerate(result.suggestions, start=1):
            print(f"  {index}. {suggestion.label}")
        return

    if result.status == "insufficient_context":
        print("\nKhông đủ context để generation.")
        return

    if result.answer:
        print("\nANSWER")
        print("-" * 92)
        print(result.answer)

    print("\nEVIDENCE MAP")
    for evidence in result.evidence:
        ref = f"Điều {evidence.article_number}" if evidence.article_number else evidence.chunk_id
        if evidence.clause_number:
            ref += f", khoản {evidence.clause_number}"
        title = f" — {evidence.article_title}" if evidence.article_title else ""
        print(f"  [{evidence.evidence_id}] score={evidence.score:.6f} {ref}{title}")


def run_query(pipeline, query: str, output_log: Path | None) -> None:
    result = pipeline.run(query, generate=True)
    print_result(result)
    append_log(output_log, result)


def main() -> None:
    args = parse_args()
    load_environment()
    config = runtime_config_from_args(
        args,
        project_root=PROJECT_ROOT,
        enable_generation=True,
    )
    pipeline, embedder, collection = build_application_pipeline(config)
    print_startup(config, embedder, collection, args.output_log)

    if args.query:
        run_query(pipeline, args.query.strip(), args.output_log)
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
                run_query(pipeline, query, args.output_log)
            except Exception as exc:
                print(f"\nERROR: {type(exc).__name__}: {exc}\n", file=sys.stderr)
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()

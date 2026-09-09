"""CLI script for chunking parsed Vietnamese labor-law documents.

Input:
    Parsed hierarchical JSON produced by LaborLawParser.

Output:
    Flat JSONL file containing one chunk per line.

Strategies:
    basic  -> one Article = one chunk
    legal  -> short Article = one chunk; long Article = split by Clause

python scripts\chunk_labor_law.py `
    -i data/extracted/labor_law.json `
    --strategy legal `
    --max-tokens 450 `
"""  

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# Allow running directly from the project without requiring `pip install -e .`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from labor_rag.processing.chunkers.basic_chunker import BasicChunker
from labor_rag.processing.chunkers.legal_structure_chunker import (
    LegalStructureChunker,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chunk parsed Vietnamese labor-law JSON into flat JSONL records."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to parsed document JSON.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path. Default: "
            "data/processed/chunks/<input_stem>_<strategy>.jsonl"
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=("basic", "legal"),
        default="legal",
        help=(
            "Chunking strategy: 'basic' keeps one Article per chunk; "
            "'legal' splits long Articles by Clause. Default: legal."
        ),
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=450,
        help=(
            "Preferred maximum chunk size for the legal strategy. "
            "Default: 450."
        ),
    )
    parser.add_argument(
        "--tokenizer-model",
        type=str,
        default=None,
        help=(
            "Optional Hugging Face model/tokenizer ID used for exact token "
            "counting, e.g. BAAI/bge-m3. If omitted, the chunker uses its "
            "dependency-free approximate token counter."
        ),
    )

    return parser.parse_args()


def load_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("Parsed document JSON must contain a root object.")

    return data


def create_token_counter(tokenizer_model: str | None):
    """Create an exact token counter from a Hugging Face tokenizer.

    If ``tokenizer_model`` is omitted, return ``None`` so BaseChunker falls
    back to its built-in approximate counter.
    """

    if not tokenizer_model:
        return None

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "--tokenizer-model requires the 'transformers' package. "
            "Install it with: pip install transformers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    def count_tokens(text: str) -> int:
        return len(
            tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=False,
            )
        )

    return count_tokens


def create_chunker(
    strategy: str,
    max_tokens: int,
    token_counter=None,
) -> BasicChunker | LegalStructureChunker:
    if strategy == "basic":
        return BasicChunker(token_counter=token_counter)

    return LegalStructureChunker(
        max_tokens=max_tokens,
        token_counter=token_counter,
    )


def default_output_path(input_path: Path, strategy: str) -> Path:
    output_dir = PROJECT_ROOT / "data" / "processed" / "chunks"
    return output_dir / f"{input_path.stem}_{strategy}.jsonl"

def write_pretty_json(
    chunks: list[dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )
        
def write_jsonl(chunks: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")


def print_summary(
    chunks: list[dict[str, Any]],
    strategy: str,
    output_path: Path,
    tokenizer_model: str | None,
) -> None:
    level_counts = Counter(
        str(chunk.get("chunk_level") or "unknown")
        for chunk in chunks
    )

    token_counts = [
        int(chunk.get("token_count") or 0)
        for chunk in chunks
    ]

    oversized_count = sum(
        bool(chunk.get("is_oversized"))
        for chunk in chunks
    )

    average_tokens = (
        sum(token_counts) / len(token_counts)
        if token_counts
        else 0.0
    )
    max_tokens = max(token_counts, default=0)

    print("Chunking completed.")
    print(f"Strategy       : {strategy}")
    print(
        "Token counter  : "
        + (
            tokenizer_model
            if tokenizer_model
            else "approximate (built-in)"
        )
    )
    print(f"Total chunks   : {len(chunks)}")
    print(
        "Chunk levels   : "
        + ", ".join(
            f"{level}={count}"
            for level, count in sorted(level_counts.items())
        )
    )
    print(f"Average tokens : {average_tokens:.1f}")
    print(f"Max tokens     : {max_tokens}")

    if strategy == "legal":
        print(f"Oversized      : {oversized_count}")

    print(f"Output         : {output_path}")


def main() -> None:
    args = parse_args()

    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be greater than 0.")

    document = load_document(args.input)

    token_counter = create_token_counter(args.tokenizer_model)

    chunker = create_chunker(
        strategy=args.strategy,
        max_tokens=args.max_tokens,
        token_counter=token_counter,
    )
    chunks = chunker.chunk(document)

    output_path = (
        args.output
        if args.output is not None
        else default_output_path(args.input, args.strategy)
    )

    write_jsonl(chunks, output_path)
    
    pretty_path = output_path.with_suffix(".pretty.json")
    write_pretty_json( chunks, pretty_path, )
    print_summary(
        chunks=chunks,
        strategy=args.strategy,
        output_path=output_path,
        tokenizer_model=args.tokenizer_model,
    )


if __name__ == "__main__":
    main()

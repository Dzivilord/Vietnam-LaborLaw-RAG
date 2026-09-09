from __future__ import annotations

import argparse
import json
from pathlib import Path

from labor_rag.processing.parsers.parser import LaborLawParser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse cleaned Vietnamese labor-law text into "
            "Document -> Chapter -> Section -> Article -> Clause JSON."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the cleaned UTF-8 TXT input file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON file or directory. Default: "
            "data/extracted/parsed_documents/<input_stem>.json"
        ),
    )
    parser.add_argument(
        "--document-id",
        default="bo-luat-lao-dong-2019",
        help="Stable document identifier.",
    )
    parser.add_argument(
        "--document-title",
        default="Bộ luật Lao động 2019",
        help="Human-readable document title.",
    )
    parser.add_argument(
        "--document-number",
        default=None,
        help="Official document number, for example 45/2019/QH14.",
    )
    parser.add_argument(
        "--document-type",
        default="Bộ luật",
        help="Document type.",
    )
    parser.add_argument(
        "--issued-date",
        default=None,
        help="Issued date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--effective-date",
        default="2021-01-01",
        help="Effective date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--source-file",
        default=None,
        help="Original source filename. Defaults to the TXT input filename.",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Write compact JSON instead of indented JSON.",
    )
    return parser.parse_args()


def build_output_path(input_path: Path, output_path: Path | None) -> Path:
    if output_path is None:
        return (
            Path("data/extracted/parsed_documents")
            / f"{input_path.stem}.json"
        )

    if output_path.exists() and output_path.is_dir():
        return output_path / f"{input_path.stem}.json"

    if output_path.suffix == "":
        return output_path / f"{input_path.stem}.json"

    if output_path.suffix.lower() != ".json":
        raise ValueError("Output path must be a .json file or a directory.")

    return output_path


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")
    if input_path.suffix.lower() != ".txt":
        raise ValueError(f"Input file must be .txt: {input_path}")

    output_path = build_output_path(args.input, args.output).resolve()
    text = input_path.read_text(encoding="utf-8")

    metadata = {
        "document_id": args.document_id,
        "document_title": args.document_title,
        "document_number": args.document_number,
        "document_type": args.document_type,
        "issued_date": args.issued_date,
        "effective_date": args.effective_date,
        "source_file": args.source_file or input_path.name,
    }

    parser = LaborLawParser()
    document = parser.parse(text, document_metadata=metadata)
    stats = parser.summarize(document)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            document,
            ensure_ascii=False,
            indent=None if args.compact else 2,
        ),
        encoding="utf-8",
    )

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Chapters: {stats['chapters']}")
    print(f"Sections: {stats['sections']}")
    print(f"Articles: {stats['articles']}")
    print(f"Clauses: {stats['clauses']}")
    print(
        "Articles without numbered clauses: "
        f"{stats['articles_without_clauses']}"
    )


if __name__ == "__main__":
    main()
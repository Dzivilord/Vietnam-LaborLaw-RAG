# scripts/clean_text.py

import argparse
from pathlib import Path

from labor_rag.ingestion.cleaners.text_cleaner import TextCleaner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Làm sạch văn bản được trích xuất từ PDF."
    )

    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="File TXT đầu vào.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="File TXT đầu ra.",
    )

    return parser.parse_args()


def build_output_path(
    input_path: Path,
    output_path: Path | None,
) -> Path:
    if output_path is not None:
        if output_path.suffix == "":
            return output_path / f"{input_path.stem}_cleaned.txt"

        return output_path

    return (
        Path("data/extracted")
        / f"{input_path.stem}_cleaned.txt"
    )


def main() -> None:
    args = parse_args()

    input_path: Path = args.input.resolve()
    output_path = build_output_path(
        input_path=input_path,
        output_path=args.output,
    ).resolve()

    raw_text = input_path.read_text(encoding="utf-8")

    cleaner = TextCleaner()
    cleaned_text = cleaner.clean(raw_text)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        cleaned_text,
        encoding="utf-8",
    )

    print(f"Đầu vào: {input_path}")
    print(f"Số ký tự ban đầu: {len(raw_text):,}")
    print(f"Số ký tự sau clean: {len(cleaned_text):,}")
    print(f"Đã lưu tại: {output_path}")


if __name__ == "__main__":
    main()
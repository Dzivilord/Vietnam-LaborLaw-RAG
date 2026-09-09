# scripts/load_pdf.py

import argparse
from pathlib import Path

from labor_rag.ingestion.loaders.pdf_loader import PDFLoader


def parse_args() -> argparse.Namespace:
    """Đọc các tham số được truyền từ command line."""

    parser = argparse.ArgumentParser( description="Trích xuất nội dung từ PDF và lưu thành file TXT." )

    parser.add_argument( "--input", "-i", type=Path, required=True, help="Đường dẫn đến file PDF đầu vào.", )

    parser.add_argument( "--output", "-o", type=Path, default=None,
        help=(
            "Đường dẫn file TXT đầu ra. "
            "Nếu không truyền, file sẽ được lưu trong "
            "data/extracted."
        ),
    )

    parser.add_argument( "--encoding", type=str, default="utf-8", help="Encoding của file đầu ra. Mặc định: utf-8.", )

    parser.add_argument( "--without-page-markers", action="store_true", help="Không thêm dấu mốc số trang vào file TXT.", )

    return parser.parse_args()


def build_output_path(
    input_path: Path,
    output_path: Path | None,
) -> Path:
    """Tạo đường dẫn đầu ra mặc định nếu người dùng không truyền --output."""

    if output_path is not None:
        return output_path

    return (
        Path("data/extracted")
        / f"{input_path.stem}.txt"
    )


def build_full_text(
    documents: list,
    include_page_markers: bool = True,
) -> str:
    """Ghép nội dung các trang thành một chuỗi hoàn chỉnh."""

    pages: list[str] = []

    for document in documents:
        content = document.content.strip()

        if not content:
            continue

        if include_page_markers:
            page_number = document.metadata.get(
                "page_number",
                "unknown",
            )
            content = (
                f"===== PAGE {page_number} =====\n"
                f"{content}"
            )

        pages.append(content)

    if include_page_markers:
        return "\n\n".join(pages)

    # Không có marker: chỉ xuống dòng một lần giữa hai trang.
    return "\n".join(pages)


def main() -> None:
    args = parse_args()

    input_path: Path = args.input
    output_path = build_output_path(
        input_path=input_path,
        output_path=args.output,
    )

    loader = PDFLoader()
    documents = loader.load(input_path)

    if not documents:
        print(f"Không trích xuất được nội dung từ: {input_path}")
        return

    full_text = build_full_text(
        documents=documents,
        include_page_markers=not args.without_page_markers,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        full_text,
        encoding=args.encoding,
    )

    print(f"File đầu vào: {input_path}")
    print(f"Tổng số trang PDF: {documents[0].metadata.get('total_pages')}")
    print(f"Số trang lấy được nội dung: {len(documents)}")
    print(f"Số ký tự: {len(full_text)}")
    print(f"Đã lưu tại: {output_path}")


if __name__ == "__main__":
    main()
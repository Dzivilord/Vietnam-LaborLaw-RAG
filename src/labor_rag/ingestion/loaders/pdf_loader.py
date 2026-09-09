
from pathlib import Path
from pypdf import PdfReader
from labor_rag.models.document import RawDocument

class PDFLoader:
    """Đọc PDF và trả về một RawDocument cho mỗi trang."""

    def load(self, file_path: Path) -> list[RawDocument]:
        self._validate_file(file_path)

        reader = PdfReader(file_path)
        documents: list[RawDocument] = []

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = self._normalize_text(text)

            if not text:
                continue

            documents.append(
                RawDocument(
                    document_id=f"{file_path.stem}-page-{page_number}",
                    content=text,
                    source_type="pdf",
                    source_path=file_path,
                    metadata={
                        "file_name": file_path.name,
                        "page_number": page_number,
                        "total_pages": len(reader.pages),
                    },
                )
            )

        return documents

    @staticmethod
    def _validate_file(file_path: Path) -> None:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy file: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Đường dẫn không phải file: {file_path}"
            )

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(
                f"File không phải PDF: {file_path}"
            )

    @staticmethod
    def _normalize_text(text: str) -> str:
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        return "\n".join(lines)
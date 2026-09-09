# # src/labor_rag/ingestion/loaders/html_loader.py

# from pathlib import Path

# from bs4 import BeautifulSoup

# from ingestion.loaders.base import BaseLoader
# from models.document import RawDocument


# class HTMLLoader(BaseLoader):
#     """Đọc file HTML cục bộ và trích xuất nội dung văn bản."""

#     def __init__(self, parser: str = "html5lib") -> None:
#         self.parser = parser

#     def load(self, source_path: Path) -> list[RawDocument]:
#         self._validate_path(source_path)

#         html = source_path.read_text(
#             encoding="utf-8",
#             errors="replace",
#         )

#         soup = BeautifulSoup(html, self.parser)

#         self._remove_unwanted_tags(soup)

#         title = self._extract_title(soup)
#         content = self._extract_text(soup)

#         if not content:
#             return []

#         document = RawDocument(
#             document_id=source_path.stem,
#             content=content,
#             source_type="html",
#             source_path=source_path,
#             metadata={
#                 "file_name": source_path.name,
#                 "title": title,
#                 "encoding": "utf-8",
#             },
#         )

#         return [document]

#     @staticmethod
#     def _validate_path(source_path: Path) -> None:
#         if not source_path.exists():
#             raise FileNotFoundError(
#                 f"Không tìm thấy file HTML: {source_path}"
#             )

#         if not source_path.is_file():
#             raise ValueError(
#                 f"Đường dẫn không phải file: {source_path}"
#             )

#         if source_path.suffix.lower() not in {".html", ".htm"}:
#             raise ValueError(
#                 f"File không có định dạng HTML: {source_path}"
#             )

#     @staticmethod
#     def _remove_unwanted_tags(soup: BeautifulSoup) -> None:
#         unwanted_tags = [
#             "script",
#             "style",
#             "noscript",
#             "svg",
#             "canvas",
#             "iframe",
#             "template",
#         ]

#         for tag in soup.find_all(unwanted_tags):
#             tag.decompose()

#     @staticmethod
#     def _extract_title(soup: BeautifulSoup) -> str | None:
#         if soup.title is None:
#             return None

#         title = soup.title.get_text(
#             separator=" ",
#             strip=True,
#         )

#         return title or None

#     @staticmethod
#     def _extract_text(soup: BeautifulSoup) -> str:
#         root = soup.body or soup

#         text = root.get_text(
#             separator="\n",
#             strip=True,
#         )

#         lines = [
#             line.strip()
#             for line in text.splitlines()
#             if line.strip()
#         ]

#         return "\n".join(lines)
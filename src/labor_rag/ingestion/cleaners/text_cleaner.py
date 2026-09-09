# src/labor_rag/ingestion/cleaners/text_cleaner.py

import re


class TextCleaner:
    """Chuẩn hóa văn bản được trích xuất từ PDF."""

    def clean(self, text: str) -> str:
        text = self._normalize_unicode_spaces(text)
        text = self._remove_page_markers(text)
        text = self._remove_repeated_whitespace(text)
        text = self._remove_empty_lines(text)

        return text.strip()

    @staticmethod
    def _normalize_unicode_spaces(text: str) -> str:
        """Chuẩn hóa các loại khoảng trắng đặc biệt."""

        replacements = {
            "\u00a0": " ",   # Non-breaking space
            "\u200b": "",    # Zero-width space
            "\ufeff": "",    # BOM
            "\t": " ",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    @staticmethod
    def _remove_page_markers(text: str) -> str:
        """Loại marker do script PDF thêm vào."""

        return re.sub(
            r"^===== PAGE \d+ =====\s*$",
            "",
            text,
            flags=re.MULTILINE,
        )

    @staticmethod
    def _remove_repeated_whitespace(text: str) -> str:
        """Loại khoảng trắng lặp bên trong từng dòng."""

        lines = []

        for line in text.splitlines():
            normalized_line = re.sub(r"[ ]{2,}", " ", line).strip()
            lines.append(normalized_line)

        return "\n".join(lines)

    @staticmethod
    def _remove_empty_lines(text: str) -> str:
        """Không cho xuất hiện nhiều dòng trống liên tiếp."""

        return re.sub(r"\n{3,}", "\n\n", text)
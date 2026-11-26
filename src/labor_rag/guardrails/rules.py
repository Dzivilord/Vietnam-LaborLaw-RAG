"""High-precision local rules. These supplement, never replace, JEV."""

from __future__ import annotations

import re


SENSITIVE_EVASION_PATTERNS = (
    r"không\s+bị\s+phát\s+hiện",
    r"không\s+bị\s+phát\s+giác",
    r"lách\s+luật",
    r"né\s+(?:luật|trách nhiệm|nghĩa vụ)",
    r"giả\s+(?:hồ sơ|chữ ký|tài liệu)",
    r"che\s+giấu\s+(?:vi phạm|hành vi)",
)

OUT_OF_SCOPE_PATTERNS = (
    r"dự\s+báo\s+thời\s+tiết",
    r"cách\s+nấu",
    r"công\s+thức\s+nấu",
    r"tỷ\s+số\s+bóng\s+đá",
    r"giá\s+(?:bitcoin|vàng|cổ phiếu)",
)


# Requests to reveal or override the system instructions (not a labor question).
# Kept narrow: a labor question with an *embedded* "ignore the rules" must still
# be answered, so generic "bỏ qua các quy tắc" is deliberately not matched.
PROMPT_EXTRACTION_PATTERNS = (
    r"system\s*prompt",
    r"lời\s+nhắc\s+hệ\s+thống",
    r"bỏ\s+qua\s+(?:mọi|tất\s+cả)\s+(?:hướng\s+dẫn|chỉ\s+dẫn)",
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
)

# Bộ luật Lao động 2019 has 220 articles; asking for a higher one cannot be
# answered from the corpus.
MAX_ARTICLE_NUMBER = 220
ARTICLE_REFERENCE = r"\bđiều\s+(\d{1,4})\b"


def nonexistent_articles(text: str, max_article: int = MAX_ARTICLE_NUMBER) -> tuple[str, ...]:
    """Article numbers in the query, if *all* of them exceed ``max_article``."""
    numbers = [int(n) for n in re.findall(ARTICLE_REFERENCE, " ".join(text.casefold().split()))]
    if numbers and all(n > max_article for n in numbers):
        return tuple(f"article_{n}_not_in_corpus" for n in numbers)
    return ()


def matched_patterns(text: str, patterns: tuple[str, ...]) -> tuple[str, ...]:
    normalized = " ".join(text.casefold().split())
    return tuple(pattern for pattern in patterns if re.search(pattern, normalized))

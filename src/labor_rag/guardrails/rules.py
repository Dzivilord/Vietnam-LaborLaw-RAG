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


def matched_patterns(text: str, patterns: tuple[str, ...]) -> tuple[str, ...]:
    normalized = " ".join(text.casefold().split())
    return tuple(pattern for pattern in patterns if re.search(pattern, normalized))

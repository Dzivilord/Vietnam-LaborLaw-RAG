from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RawDocument:
    """Dữ liệu thô được trích xuất từ một nguồn tài liệu."""

    document_id: str
    content: str
    source_type: str
    source_path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
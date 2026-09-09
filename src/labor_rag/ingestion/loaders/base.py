from abc import ABC, abstractmethod
from pathlib import Path

from labor_rag.models.document import RawDocument


class BaseLoader(ABC):
    @abstractmethod
    def load(self, file_path: Path) -> list[RawDocument]:
        """Đọc nguồn dữ liệu và trả về các tài liệu thô."""
        raise NotImplementedError
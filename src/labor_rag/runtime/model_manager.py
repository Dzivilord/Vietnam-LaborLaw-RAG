"""Runtime model/component lifecycle management."""

from __future__ import annotations

from .config import RAGRuntimeConfig
from .factory import build_application_pipeline


class ModelManager:
    """Create and keep RAG components alive.

    In production this object is initialized once when the server starts.
    """

    def __init__(self, config: RAGRuntimeConfig):
        self.config = config
        self.pipeline = None
        self.embedder = None
        self.collection = None

    def startup(self):
        (
            self.pipeline,
            self.embedder,
            self.collection,
        ) = build_application_pipeline(self.config)

        return self

    def shutdown(self):
        self.pipeline = None
        self.embedder = None

    def ready(self) -> bool:
        return self.pipeline is not None

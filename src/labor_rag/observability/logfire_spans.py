"""Optional Logfire tracing without coupling the RAG core to Logfire.

Conventions (keep them stable: dashboards and SQL queries depend on them):

* Span names are dotted and low-cardinality: ``rag.request``, ``rag.guardrail.input``,
  ``rag.retrieve``, ``rag.rerank``, ``rag.context``, ``rag.generate`` and, for
  evaluation runs, ``eval.run`` / ``eval.case`` / ``eval.summary``.
* Attributes are namespaced: ``rag.*`` for pipeline facts, ``eval.*`` for
  evaluation facts.  IDs (chunk IDs, case IDs) are fine; user text is only
  attached when :func:`content_allowed` is true.
* Results are attached *after* the work finishes via ``handle.set_attribute``
  so one span carries both inputs and outcome (filterable in Logfire).

Tracing is off unless ``LOGFIRE_ENABLED=true``; every helper then degrades to
a no-op so tests and offline runs need neither the package nor a token.
"""

from __future__ import annotations

from contextlib import nullcontext
import os
from typing import Any, ContextManager, Iterable


class _NoopSpan:
    """Stands in for a Logfire span when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def set_attributes(self, attributes: dict[str, Any]) -> None:
        return None

    def record_exception(self, exception: BaseException, **_: Any) -> None:
        return None


NOOP_SPAN = _NoopSpan()


def enabled() -> bool:
    return os.getenv("LOGFIRE_ENABLED", "false").lower() == "true"


def content_allowed() -> bool:
    """Whether user text (queries, answers) may be attached to spans."""
    return os.getenv("RAG_TRACE_INCLUDE_CONTENT", "false").lower() == "true"


def span(name: str, *, tags: Iterable[str] | None = None, **attributes: Any) -> ContextManager[Any]:
    """Open a Logfire span named ``name``; yields a handle with ``set_attribute``.

    Keys may contain dots (``rag.top_k``); they are passed through unchanged.
    """
    if not enabled():
        return nullcontext(NOOP_SPAN)
    try:
        import logfire
    except ImportError:
        return nullcontext(NOOP_SPAN)
    clean = {key: _jsonable(value) for key, value in attributes.items() if value is not None}
    return logfire.span(name, _span_name=name, _tags=list(tags or ()), **clean)


def set_attributes(handle: Any, **attributes: Any) -> None:
    """Attach outcome attributes to a span handle, skipping ``None`` values."""
    handle.set_attributes({key: _jsonable(value) for key, value in attributes.items() if value is not None})


def log(message: str, *, level: str = "info", tags: Iterable[str] | None = None, **attributes: Any) -> None:
    """Emit a standalone Logfire log record (no-op when tracing is disabled)."""
    if not enabled():
        return
    try:
        import logfire
    except ImportError:
        return
    clean = {key: _jsonable(value) for key, value in attributes.items() if value is not None}
    logfire.log(level, message, attributes=clean, tags=list(tags or ()))


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)

"""Logfire setup shared by every evaluation phase.

Trace layout in Logfire (one trace per eval run):

    eval.run                      phase, run_id, git sha, config hash, models
    ├── eval.case                 one per golden case: expected vs actual + outcome
    │   ├── rag.retrieve          (pipeline spans, phase 2 only)
    │   ├── rag.rerank
    │   ├── rag.context
    │   └── rag.generate
    └── eval.summary (log)        flattened metrics: eval.metric.<group>.<name>

Filter a run with ``attributes->>'eval.run_id' = '...'``; compare runs by
``eval.config_hash`` / ``eval.phase``.  See evals/README.md for SQL queries.

Tracing is enabled when Logfire is installed and a token is available
(``LOGFIRE_TOKEN`` env var or ``logfire auth`` credentials); disable with
``EVAL_LOGFIRE=false``.  Golden questions are synthetic, so query/answer text is
attached by default here (``RAG_TRACE_INCLUDE_CONTENT`` overrides it).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from labor_rag.observability import logfire_spans

SERVICE_NAME = "labor-rag-eval"


def environment() -> str:
    if os.getenv("EVAL_ENVIRONMENT"):
        return os.environ["EVAL_ENVIRONMENT"]
    return "kaggle" if os.getenv("KAGGLE_KERNEL_RUN_TYPE") else "local"


def _git_sha() -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return f"{sha}{'-dirty' if dirty else ''}" if sha else None


def config_hash(path: str | Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:10]
    except OSError:
        return "unknown"


def run_metadata(phase: str, *, config_path: str | Path | None = None, **extra: Any) -> dict[str, Any]:
    """Identity of one eval run; stored in the report and on the root span."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    chash = config_hash(config_path) if config_path else "none"
    return {
        "run_id": f"{phase}-{timestamp}-{chash[:6]}",
        "phase": phase,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
        "git_sha": _git_sha(),
        "config_path": str(config_path) if config_path else None,
        "config_hash": chash,
        **extra,
    }


def configure() -> bool:
    """Configure Logfire for an eval process. Returns True when tracing is on."""
    if os.getenv("EVAL_LOGFIRE", "true").lower() == "false":
        return False
    try:
        import logfire
    except ImportError:
        return False
    os.environ.setdefault("RAG_TRACE_INCLUDE_CONTENT", "true")
    logfire.configure(
        service_name=SERVICE_NAME,
        environment=environment(),
        send_to_logfire="if-token-present",
        console=False,
    )
    os.environ["LOGFIRE_ENABLED"] = "true"
    return True


def flush() -> None:
    if not logfire_spans.enabled():
        return
    try:
        import logfire

        logfire.force_flush()
    except Exception:  # noqa: BLE001 - never fail an eval because telemetry could not flush
        pass


@contextmanager
def run_span(meta: dict[str, Any]) -> Iterator[Any]:
    attributes = {f"eval.{key}": value for key, value in meta.items()}
    with logfire_spans.span("eval.run", tags=[meta["phase"]], **attributes) as handle:
        yield handle


@contextmanager
def case_span(meta: dict[str, Any], case: dict[str, Any]) -> Iterator[Any]:
    attributes = {
        "eval.run_id": meta["run_id"],
        "eval.phase": meta["phase"],
        "eval.case_id": case["id"],
        "eval.category": case.get("category"),
        "eval.subcategory": case.get("subcategory"),
        "eval.split": case.get("split"),
        "eval.expected_route": case.get("expected_route"),
        "eval.expected_action": case.get("expected_action"),
    }
    if logfire_spans.content_allowed():
        attributes["eval.query"] = case.get("query")
    with logfire_spans.span("eval.case", tags=[meta["phase"], str(case.get("category"))], **attributes) as handle:
        yield handle


def set_outcome(handle: Any, **attributes: Any) -> None:
    logfire_spans.set_attributes(handle, **{f"eval.{key}": value for key, value in attributes.items()})


def flatten_metrics(metrics: dict[str, Any], prefix: str = "eval.metric") -> dict[str, float | int]:
    """``{"retrieval": {"hit@5": 0.9}}`` -> ``{"eval.metric.retrieval.hit@5": 0.9}`` (numbers only)."""
    flat: dict[str, float | int] = {}
    for key, value in metrics.items():
        name = f"{prefix}.{key}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            flat[name] = value
    return flat


def log_summary(meta: dict[str, Any], metrics: dict[str, Any], run_handle: Any = None) -> None:
    """Emit ``eval.summary`` and copy headline numbers onto the run span."""
    flat = flatten_metrics(metrics)
    logfire_spans.log("eval.summary", tags=[meta["phase"]], **{"eval.run_id": meta["run_id"], "eval.phase": meta["phase"], **flat})
    if run_handle is not None:
        logfire_spans.set_attributes(run_handle, **flat)

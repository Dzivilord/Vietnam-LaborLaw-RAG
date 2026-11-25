"""Small, dependency-light request tracing for the admin console.

The in-memory store is intentionally useful for local development.  Firestore
is an optional durable sink selected through environment variables; it is not a
substitute for production authentication or a full observability platform.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any


@dataclass(frozen=True, slots=True)
class RequestTrace:
    request_id: str
    created_at: str
    query: str | None
    query_hash: str
    status: str
    total_latency_ms: float
    timings_ms: dict[str, float]
    retrieved_count: int | None
    reranked_count: int | None
    evidence_count: int
    citations: list[str]
    provider: str | None
    model: str | None
    error: str | None = None


class RequestTraceRecorder:
    """Keep recent traces locally and optionally mirror privacy-safe fields."""

    def __init__(self, max_entries: int = 200) -> None:
        self._traces: deque[RequestTrace] = deque(maxlen=max_entries)
        self._lock = Lock()
        self._firestore_collection = os.getenv("FIRESTORE_TRACE_COLLECTION")
        self._include_content = os.getenv("RAG_TRACE_INCLUDE_CONTENT", "false").lower() == "true"
        self._firestore = self._create_firestore_client()
        self._live_events: dict[str, list[dict[str, Any]]] = {}
        self._finished_runs: set[str] = set()

    def record(
        self,
        *,
        request_id: str,
        query: str,
        status: str,
        total_latency_ms: float,
        diagnostics: dict[str, Any],
        evidence_count: int,
        citations: list[str],
        error: str | None = None,
    ) -> RequestTrace:
        pipeline = diagnostics.get("pipeline", {})
        generation = diagnostics.get("generation", {})
        trace = RequestTrace(
            request_id=request_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            query=query,
            query_hash=hashlib.sha256(query.encode("utf-8")).hexdigest()[:16],
            status=status,
            total_latency_ms=round(total_latency_ms, 2),
            timings_ms=dict(pipeline.get("timings_ms", {})),
            retrieved_count=pipeline.get("retrieved_count"),
            reranked_count=pipeline.get("reranked_count"),
            evidence_count=evidence_count,
            citations=list(citations),
            provider=generation.get("provider"),
            model=generation.get("model"),
            error=error,
        )
        with self._lock:
            self._traces.appendleft(trace)
        self._write_firestore(trace)
        return trace

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        safe_limit = min(max(limit, 1), 200)
        with self._lock:
            return [asdict(item) for item in list(self._traces)[:safe_limit]]

    def start_live_run(self, request_id: str) -> None:
        with self._lock:
            self._live_events[request_id] = []
            self._finished_runs.discard(request_id)
        self.emit_live_event(request_id, "queued", "started", "Request accepted; retrieval is next.")

    def emit_live_event(self, request_id: str, stage: str, status: str, message: str, **data: Any) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "status": status,
            "message": message,
            **data,
        }
        with self._lock:
            events = self._live_events.setdefault(request_id, [])
            events.append(event)
            if len(events) > 100:
                del events[:-100]

    def live_events(self, request_id: str, after: int = 0) -> tuple[list[dict[str, Any]], bool, int]:
        with self._lock:
            events = self._live_events.get(request_id, [])
            return list(events[after:]), request_id in self._finished_runs, len(events)

    def finish_live_run(self, request_id: str) -> None:
        with self._lock:
            self._finished_runs.add(request_id)

    def _create_firestore_client(self):
        if not self._firestore_collection:
            return None
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore

            if not firebase_admin._apps:
                credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                credential = credentials.Certificate(credential_path) if credential_path else None
                firebase_admin.initialize_app(credential)
            return firestore.client()
        except Exception:
            # Trace recording must never break an answer request. Surface sink
            # health through infrastructure logs/metrics in production.
            return None

    def _write_firestore(self, trace: RequestTrace) -> None:
        if self._firestore is None or not self._firestore_collection:
            return
        try:
            payload = asdict(trace)
            if not self._include_content:
                payload["query"] = None
            self._firestore.collection(self._firestore_collection).document(trace.request_id).set(payload)
        except Exception:
            return

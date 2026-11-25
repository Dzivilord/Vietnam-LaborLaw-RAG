import json
import threading
import time
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

from app.backend.api.schemas import ChatRequest, ChatResponse, EvidenceCitation

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/traces")
def recent_traces(request: Request, limit: int = Query(default=50, ge=1, le=200)):
    """Return recent local request traces for a protected admin deployment."""
    return {"traces": request.app.state.trace_recorder.recent(limit)}


@router.post("/runs")
def start_live_run(body: ChatRequest, request: Request):
    """Start a diagnostic run and return its SSE stream identifier immediately."""
    request_id = str(uuid4())
    recorder = request.app.state.trace_recorder
    recorder.start_live_run(request_id)
    runtime = request.app.state.rag_runtime

    def execute() -> None:
        started = perf_counter()
        try:
            options = body.options
            result = runtime.chat(
                body.query,
                enable_rerank=options.enable_rerank if options else None,
                generation_overrides=(options.generation.model_dump(exclude_none=True, exclude_defaults=True) if options else {}),
                event_callback=lambda stage, status, message: recorder.emit_live_event(request_id, stage, status, message),
            )
            diagnostics = result.metadata.get("diagnostics", {})
            recorder.record(
                request_id=request_id, query=body.query, status=result.metadata.get("status", "error"),
                total_latency_ms=(perf_counter() - started) * 1000.0, diagnostics=diagnostics,
                evidence_count=len(result.evidence), citations=result.metadata.get("citations", []), error=result.metadata.get("error"),
            )
            payload = ChatResponse(
                request_id=request_id, answer=result.answer, citations=result.metadata.get("citations", []),
                evidence=[EvidenceCitation(evidence_id=item.evidence_id, chunk_id=item.chunk_id, content=item.content, document_title=item.document_title, document_number=item.document_number, source_file=item.source_file, article_number=item.article_number, article_title=item.article_title, clause_number=item.clause_number, point_number=item.point_number) for item in result.evidence],
                debug=result.metadata,
            ).model_dump(mode="json")
            if result.metadata.get("status") == "error":
                recorder.emit_live_event(
                    request_id, "error", "failed", "Pipeline stopped with an error.",
                    error=result.metadata.get("error", "Unknown runtime error"), response=payload,
                )
            else:
                recorder.emit_live_event(request_id, "completed", "completed", "Chat cycle completed successfully.", response=payload)
        except Exception as exc:
            recorder.emit_live_event(request_id, "error", "failed", "Chat cycle failed.", error=str(exc))
        finally:
            recorder.finish_live_run(request_id)

    threading.Thread(target=execute, daemon=True).start()
    return {"request_id": request_id, "events_url": f"/admin/runs/{request_id}/events"}


@router.get("/runs/{request_id}/events")
def stream_live_events(request_id: str, request: Request):
    recorder = request.app.state.trace_recorder

    def stream():
        cursor = 0
        while True:
            events, finished, cursor = recorder.live_events(request_id, cursor)
            for event in events:
                yield f"event: trace\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            if finished:
                yield "event: end\ndata: {}\n\n"
                return
            time.sleep(0.2)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

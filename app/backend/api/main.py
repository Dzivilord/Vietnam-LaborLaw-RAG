from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.api.routes.admin import router as admin_router
from app.backend.api.routes.chat import router as chat_router
from labor_rag.observability.request_traces import RequestTraceRecorder
from labor_rag.runtime.config import RAGRuntimeConfig
from labor_rag.runtime.model_manager import ModelManager
from labor_rag.runtime.rag_runtime import RAGRuntime


@asynccontextmanager
async def lifespan(app: FastAPI):
    project_root = Path(__file__).resolve().parents[3]
    try:
        from dotenv import load_dotenv
        load_dotenv(project_root / ".env", override=False)
    except ImportError:
        pass
    config = RAGRuntimeConfig.from_yaml(project_root / "configs" / "labor_rag.yaml")

    manager = ModelManager(config)
    manager.startup()

    app.state.model_manager = manager
    app.state.rag_runtime = RAGRuntime(
        pipeline=manager.pipeline,
        config=config,
    )
    app.state.trace_recorder = RequestTraceRecorder()

    yield

    manager.shutdown()


app = FastAPI(title="Labor Law RAG API", version="0.1.0", lifespan=lifespan)

# Hosted Logfire is opt-in. The default local mode emits no data outside this
# machine; set LOGFIRE_ENABLED=true deliberately when hosted tracing is wanted.
if os.getenv("LOGFIRE_ENABLED", "false").lower() == "true":
    try:
        import logfire

        logfire.configure(service_name="labor-law-rag", send_to_logfire="if-token-present")
        os.environ.setdefault("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS", "/health,/admin/traces,/admin/runs/.*/events")
        logfire.instrument_fastapi(app, extra_spans=True)
    except ImportError:
        pass
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)
app.include_router(admin_router)


@app.get("/health")
def health():
    return {"status": "ok"}

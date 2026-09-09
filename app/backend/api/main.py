from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.backend.api.routes.chat import router as chat_router
from labor_rag.runtime.config import RAGRuntimeConfig
from labor_rag.runtime.model_manager import ModelManager
from labor_rag.runtime.rag_runtime import RAGRuntime


@asynccontextmanager
async def lifespan(app: FastAPI):
    project_root = Path(__file__).resolve().parents[3]
    config = RAGRuntimeConfig.from_yaml(project_root / "configs" / "labor_rag.yaml")

    manager = ModelManager(config)
    manager.startup()

    app.state.model_manager = manager
    app.state.rag_runtime = RAGRuntime(
        pipeline=manager.pipeline,
        config=config,
    )

    yield

    manager.shutdown()


app = FastAPI(title="Labor Law RAG API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)


@app.get("/health")
def health():
    return {"status": "ok"}

"""Health checks used by FastAPI/Docker orchestration."""


def health_status(model_manager) -> dict:
    return {
        "status": "ok" if model_manager.ready() else "not_ready",
        "pipeline_loaded": model_manager.ready(),
    }

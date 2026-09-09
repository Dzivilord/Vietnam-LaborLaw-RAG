FastAPI backend layer.

Run from the project root (`D:\Projects\RAG`) after activating the Python
environment and indexing the chunk data into Qdrant:

```powershell
python -m uvicorn app.backend.api.main:app --reload
```

The canonical setup and startup instructions are in the root `README.md`.

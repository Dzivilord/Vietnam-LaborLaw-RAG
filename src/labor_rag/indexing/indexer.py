"""Simple indexing flow: chunks -> embeddings -> Qdrant."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def index_chunks(
    chunks: Iterable[dict[str, Any]],
    embedder: Any,
    store: Any,
    *,
    recreate_collection: bool = False,
) -> int:
    """Embed chunks and store their vectors + payloads in Qdrant.

    Expected chunk fields:
        - chunk_id
        - embedding_text

    All remaining chunk fields are stored in Qdrant as payload.

    This first version intentionally supports one dense embedding space per
    collection. Sparse/hybrid indexing can be added later after the dense flow
    is working end-to-end.
    """
    chunks = list(chunks)

    if not chunks:
        return 0

    texts: list[str] = []
    ids: list[str] = []

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        embedding_text = chunk.get("embedding_text")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("Every chunk must have a non-empty 'chunk_id'.")

        if not isinstance(embedding_text, str) or not embedding_text.strip():
            raise ValueError(
                f"Chunk {chunk_id!r} must have a non-empty 'embedding_text'."
            )

        ids.append(chunk_id)
        texts.append(embedding_text)

    vectors = embedder.embed_documents(texts)

    if len(vectors) != len(chunks):
        raise RuntimeError(
            f"Embedder returned {len(vectors)} vectors "
            f"for {len(chunks)} chunks."
        )

    store.create_collection(
        vector_size=embedder.dimension,
        recreate=recreate_collection,
    )

    store.upsert(
        ids=ids,
        vectors=vectors,
        payloads=chunks,
    )

    return len(chunks)

"""Vector store behind a `VectorStore` protocol, backed by Chroma.

Defaults to an in-memory (ephemeral) collection — ideal for tests and ephemeral
serving — or persists to disk when a directory is given. Uses cosine space so the
returned `score` is a cosine similarity in [-1, 1] (higher = more similar).

A future `VertexVectorStore` (Phase 7) implements the same protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel, ConfigDict, Field


class RetrievedChunk(BaseModel):
    """A search hit: the stored text, its similarity score, and metadata."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    score: float = Field(description="Cosine similarity in [-1, 1]; higher is more similar.")
    metadata: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    """Stores embedded chunks and retrieves nearest neighbours for a query vector."""

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None: ...

    def query(self, embedding: list[float], k: int) -> list[RetrievedChunk]: ...

    def count(self) -> int: ...

    def reset(self) -> None: ...


class ChromaVectorStore:
    """Chroma-backed vector store (in-memory by default, persistent if a dir is set)."""

    def __init__(
        self, collection_name: str = "invoices", persist_dir: Path | str | None = None
    ) -> None:
        settings = ChromaSettings(anonymized_telemetry=False, allow_reset=True)
        if persist_dir is None:
            self._client = chromadb.EphemeralClient(settings=settings)
        else:
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, str]],
    ) -> None:
        if not ids:
            return
        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[dict(m) for m in metadatas],
        )

    def query(self, embedding: list[float], k: int) -> list[RetrievedChunk]:
        result: dict[str, Any] = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]
        hits: list[RetrievedChunk] = []
        for cid, text, meta, distance in zip(ids, documents, metadatas, distances, strict=True):
            metadata = {str(key): str(value) for key, value in (meta or {}).items()}
            hits.append(
                RetrievedChunk(id=cid, text=text, score=1.0 - float(distance), metadata=metadata)
            )
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )

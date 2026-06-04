"""Embedding providers behind a single `EmbeddingProvider` protocol.

Two local implementations:
  * `SentenceTransformerEmbedder` — real semantic embeddings (all-MiniLM-L6-v2),
    the production-quality local path. Model name is pinned so every environment
    fetches the same weights.
  * `HashingEmbedder` — deterministic, dependency-free hashing-trick vectors.
    Used for fast, hermetic unit tests (no model download, no network).

A future `VertexEmbedder` (Phase 7) implements the same protocol, so swapping is
a config change, not a rewrite.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_TOKEN_RE = re.compile(r"[a-z0-9@.]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into fixed-length, L2-normalized vectors."""

    @property
    def name(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder (no external model)."""

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    @property
    def name(self) -> str:
        return "hashing"

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self._dimension
        for token in _tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()  # noqa: S324 - non-crypto bucketing
            bucket = int.from_bytes(digest[:4], "big") % self._dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class SentenceTransformerEmbedder:
    """Real semantic embeddings via sentence-transformers (lazy-loaded)."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self._model: object | None = None
        self._dimension: int | None = None

    @property
    def name(self) -> str:
        return "minilm"

    def _load(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - lazy/optional

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        # Derived from a real encode — version-agnostic across sentence-transformers.
        if self._dimension is None:
            self._dimension = len(self.embed_query("dimension probe"))
        return self._dimension

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(  # type: ignore[attr-defined]
            texts, normalize_embeddings=True, convert_to_numpy=True
        )
        return [[float(x) for x in row] for row in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]

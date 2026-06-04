"""GCP Vertex AI helpers (lazy, opt-in Phase 7 swap-ins)."""

from invoice_iq.vertex.client import VertexClassifierClient
from invoice_iq.vertex.embeddings import VertexEmbedder

__all__ = ["VertexClassifierClient", "VertexEmbedder"]

"""Retrieval-augmented Q&A over indexed invoices.

`RAGPipeline` wires an `EmbeddingProvider` to a `VectorStore`: it indexes the
field-aware chunks of each invoice, then answers a question by embedding it,
retrieving the most similar chunks, and returning an extractive answer with
citations. Answer synthesis is deliberately extractive here (no LLM, no key); the
Phase 5 agent layer adds LLM synthesis on top of these same retrieved sources.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from invoice_iq.rag.chunking import chunk_invoice
from invoice_iq.rag.embeddings import EmbeddingProvider
from invoice_iq.rag.vector_store import RetrievedChunk, VectorStore
from invoice_iq.schemas.invoice import Invoice


class Answer(BaseModel):
    """An answer plus the retrieved chunks that support it (citations)."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    sources: list[RetrievedChunk] = Field(default_factory=list)

    @property
    def top_source(self) -> RetrievedChunk | None:
        return self.sources[0] if self.sources else None


class RAGPipeline:
    """Indexes invoices and answers questions over them."""

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def index_invoice(self, invoice: Invoice, doc_id: str) -> int:
        """Chunk, embed, and store one invoice. Returns the number of chunks added."""
        chunks = chunk_invoice(invoice, doc_id)
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(
            ids=[c.id for c in chunks],
            texts=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.metadata for c in chunks],
        )
        return len(chunks)

    def index_invoices(self, items: list[tuple[str, Invoice]]) -> int:
        """Index many (doc_id, invoice) pairs. Returns total chunks added."""
        return sum(self.index_invoice(invoice, doc_id) for doc_id, invoice in items)

    def ask(self, question: str, k: int = 4) -> Answer:
        """Retrieve the most relevant chunks and return an extractive answer."""
        query_vec = self.embedder.embed_query(question)
        hits = self.store.query(query_vec, k=k)
        if not hits:
            return Answer(question=question, answer="No relevant documents were found.", sources=[])
        # Prefer a 'summary' chunk among the hits for a complete, answer-like
        # response (it carries the totals); otherwise fall back to the top hit.
        summary_hit = next((h for h in hits if h.metadata.get("kind") == "summary"), None)
        answer_text = (summary_hit or hits[0]).text
        return Answer(question=question, answer=answer_text, sources=hits)

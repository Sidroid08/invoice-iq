"""Tests for the RAG pipeline (Phase 4).

Most tests use the deterministic HashingEmbedder (fast, no model download). The
headline semantic-retrieval test uses the real all-MiniLM-L6-v2 model and is
marked `slow` (deselect with `-m "not slow"`).
"""

from __future__ import annotations

import math
from datetime import date
from decimal import Decimal

import pytest

from invoice_iq.rag import (
    ChromaVectorStore,
    HashingEmbedder,
    RAGPipeline,
    SentenceTransformerEmbedder,
    chunk_invoice,
    chunk_text,
)
from invoice_iq.rag.embeddings import EmbeddingProvider
from invoice_iq.rag.vector_store import VectorStore
from invoice_iq.schemas.invoice import Invoice, LineItem, Money, Vendor


def _invoice(vendor: str, number: str, amount: str) -> Invoice:
    money = Money(amount=amount, currency="USD")
    return Invoice(
        invoice_number=number,
        vendor=Vendor(name=vendor),
        invoice_date=date(2026, 1, 1),
        currency="USD",
        line_items=[
            LineItem(
                description="Services",
                quantity=Decimal("1"),
                unit_price=money,
                line_total=money,
            )
        ],
        subtotal=money,
        total=money,
    )


# Three vendors with distinct totals — the retrieval target set.
_INVOICES = [
    ("docA", _invoice("Acme Corp", "INV-A", "1210.00")),
    ("docG", _invoice("Globex LLC", "INV-G", "50.00")),
    ("docI", _invoice("Initech", "INV-I", "999.00")),
]


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def test_hashing_embedder_protocol_and_shape() -> None:
    emb = HashingEmbedder(dimension=128)
    assert isinstance(emb, EmbeddingProvider)
    vec = emb.embed_query("hello world")
    assert len(vec) == 128
    assert emb.dimension == 128


def test_hashing_embedder_deterministic_and_normalized() -> None:
    emb = HashingEmbedder()
    v1 = emb.embed_query("Acme Corp total 1210.00")
    v2 = emb.embed_query("Acme Corp total 1210.00")
    assert v1 == v2
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, abs_tol=1e-6)
    assert emb.embed_query("completely different text") != v1


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def test_chunk_invoice_produces_field_aware_chunks() -> None:
    _, invoice = _INVOICES[0]
    chunks = chunk_invoice(invoice, "docA")
    kinds = {c.metadata["kind"] for c in chunks}
    assert kinds == {"summary", "vendor", "line_item"}
    assert len({c.id for c in chunks}) == len(chunks)  # unique ids
    summary = next(c for c in chunks if c.metadata["kind"] == "summary")
    assert "1210.00" in summary.text
    assert summary.metadata["vendor"] == "Acme Corp"


def test_chunk_text_sliding_window() -> None:
    assert chunk_text("") == []
    words = " ".join(str(i) for i in range(100))
    chunks = chunk_text(words, size=30, overlap=10)
    assert len(chunks) > 1
    assert all(chunks)


def test_chunk_text_validates_params() -> None:
    with pytest.raises(ValueError):
        chunk_text("a b c", size=0)
    with pytest.raises(ValueError):
        chunk_text("a b c", size=10, overlap=10)


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
def test_vector_store_add_query_count_reset() -> None:
    emb = HashingEmbedder()
    store = ChromaVectorStore(collection_name="test_store")
    assert isinstance(store, VectorStore)
    texts = ["alpha invoice acme", "beta receipt globex"]
    store.add(
        ids=["1", "2"],
        texts=texts,
        embeddings=emb.embed_documents(texts),
        metadatas=[{"k": "a"}, {"k": "b"}],
    )
    assert store.count() == 2
    hits = store.query(emb.embed_query("alpha invoice acme"), k=1)
    assert len(hits) == 1 and hits[0].id == "1"
    store.reset()
    assert store.count() == 0


# --------------------------------------------------------------------------- #
# RAG pipeline — retrieval hits the right doc, answer contains the value
# --------------------------------------------------------------------------- #
def _build_pipeline(embedder: EmbeddingProvider, name: str) -> RAGPipeline:
    store = ChromaVectorStore(collection_name=name)
    store.reset()
    pipe = RAGPipeline(embedder=embedder, store=store)
    pipe.index_invoices(_INVOICES)
    return pipe


def test_rag_retrieval_hashing() -> None:
    pipe = _build_pipeline(HashingEmbedder(), "rag_hashing")
    answer = pipe.ask("What is the total for Acme Corp?")
    assert answer.sources[0].metadata["vendor"] == "Acme Corp"
    assert "1210.00" in answer.answer


@pytest.mark.slow
def test_rag_retrieval_semantic_minilm() -> None:
    """Headline test: real MiniLM embeddings retrieve the right invoice."""
    pipe = _build_pipeline(SentenceTransformerEmbedder(), "rag_minilm")
    answer = pipe.ask("How much do we owe Acme Corporation in total?")
    assert answer.sources[0].metadata["vendor"] == "Acme Corp"
    assert "1210.00" in answer.answer

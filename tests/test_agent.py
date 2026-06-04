"""Tests for the LangGraph agent (Phase 5).

The agent runs deterministically (no LLM), so these exercise the real graph:
state transitions, conditional routing, and the typed final result. The classifier
checkpoint is gitignored, so a tiny model is trained in-fixture.
"""

from __future__ import annotations

import random
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import pytest
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from invoice_iq.agent import AgentDeps, NextAction, build_agent, run_agent
from invoice_iq.agent.tools import build_structured_tools
from invoice_iq.classifier import LABELS, Predictor
from invoice_iq.classifier.train import TrainConfig, train_classifier
from invoice_iq.extraction import RuleBasedExtractor
from invoice_iq.extraction.base import ExtractionError, Extractor
from invoice_iq.rag import ChromaVectorStore, HashingEmbedder, RAGPipeline
from invoice_iq.schemas.documents import DocumentType, OCRResult
from invoice_iq.schemas.invoice import Invoice
from invoice_iq.synthetic import (
    random_contract_lines,
    render_invoice_pdf,
    render_text_pdf,
)


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    model, vocab, _ = train_classifier(
        TrainConfig(n_train_per_class=20, n_test_per_class=5, epochs=10, seed=7)
    )
    return Predictor(model=model, vocab=vocab, labels=list(LABELS))


def _deps(predictor: Predictor, **overrides: object) -> AgentDeps:
    store = ChromaVectorStore(collection_name=f"agent_{uuid4().hex[:8]}")
    return AgentDeps(
        predictor=predictor,
        extractor=RuleBasedExtractor(),
        rag=RAGPipeline(HashingEmbedder(), store),
        **overrides,  # type: ignore[arg-type]
    )


class _FailingExtractor:
    """An Extractor that always fails - exercises the error branch."""

    name = "failing"

    def extract(self, ocr: OCRResult) -> Invoice:
        raise ExtractionError("synthetic failure")


def _args_schema(tool: StructuredTool) -> type[BaseModel]:
    schema = tool.args_schema
    assert isinstance(schema, type)
    assert issubclass(schema, BaseModel)
    return schema


def test_build_agent_returns_compiled_graph(predictor: Predictor) -> None:
    app = build_agent(_deps(predictor))
    assert hasattr(app, "invoke")


def test_invoice_flow_with_question(
    tmp_path: Path, predictor: Predictor, valid_invoice: Invoice
) -> None:
    pdf = tmp_path / "inv.pdf"
    render_invoice_pdf(pdf, valid_invoice)
    result = run_agent(_deps(predictor), pdf, question="What is the total?")

    assert result.document_type is DocumentType.INVOICE
    assert result.invoice is not None
    assert result.invoice.total.amount == Decimal("1210.00")
    assert result.answer is not None
    assert "1210.00" in result.answer.answer
    assert result.recommendation is NextAction.APPROVE
    # Full path visited: classify -> extract -> answer -> recommend
    joined = " | ".join(result.trace)
    assert "classified" in joined and "extracted" in joined
    assert "answered" in joined and "recommend" in joined


def test_invoice_flow_without_question_skips_answer(
    tmp_path: Path, predictor: Predictor, valid_invoice: Invoice
) -> None:
    pdf = tmp_path / "inv.pdf"
    render_invoice_pdf(pdf, valid_invoice)
    result = run_agent(_deps(predictor), pdf)
    assert result.invoice is not None
    assert result.answer is None
    assert "answered" not in " | ".join(result.trace)
    assert result.recommendation is NextAction.APPROVE


def test_non_invoice_routes_and_skips_extraction(tmp_path: Path, predictor: Predictor) -> None:
    pdf = tmp_path / "contract.pdf"
    render_text_pdf(pdf, random_contract_lines(random.Random(1)))
    result = run_agent(_deps(predictor), pdf, question="anything?")

    assert result.document_type is DocumentType.CONTRACT
    assert result.invoice is None
    assert result.answer is None
    assert result.recommendation is NextAction.ROUTE_NON_INVOICE
    assert "extracted" not in " | ".join(result.trace)


def test_high_value_invoice_flagged_for_review(
    tmp_path: Path, predictor: Predictor, valid_invoice: Invoice
) -> None:
    pdf = tmp_path / "inv.pdf"
    render_invoice_pdf(pdf, valid_invoice)  # total 1210.00
    deps = _deps(predictor, review_total=Decimal("100"))
    result = run_agent(deps, pdf)
    assert result.recommendation is NextAction.REVIEW
    assert "High-value" in result.rationale


def test_extraction_failure_flagged_for_review(
    tmp_path: Path, predictor: Predictor, valid_invoice: Invoice
) -> None:
    pdf = tmp_path / "inv.pdf"
    render_invoice_pdf(pdf, valid_invoice)
    store = ChromaVectorStore(collection_name=f"agent_{uuid4().hex[:8]}")
    deps = AgentDeps(
        predictor=predictor,
        extractor=_FailingExtractor(),
        rag=RAGPipeline(HashingEmbedder(), store),
    )
    result = run_agent(deps, pdf)
    assert result.invoice is None
    assert result.errors
    assert result.recommendation is NextAction.REVIEW
    assert "extraction failed" in result.rationale.lower()


def test_structured_tools_exposed(predictor: Predictor) -> None:
    deps = _deps(predictor)
    tool_list = build_structured_tools(deps.predictor, deps.extractor, deps.rag)
    tools_by_name = {t.name: t for t in tool_list}
    assert set(tools_by_name) == {"classify_document", "extract_invoice", "answer_question"}

    assert _args_schema(tools_by_name["classify_document"]).model_fields[
        "ocr"
    ].annotation is OCRResult
    assert _args_schema(tools_by_name["extract_invoice"]).model_fields[
        "ocr"
    ].annotation is OCRResult

    answer_fields = _args_schema(tools_by_name["answer_question"]).model_fields
    assert answer_fields["invoice"].annotation is Invoice
    assert answer_fields["doc_id"].annotation is str
    assert answer_fields["question"].annotation is str


def test_failing_extractor_satisfies_protocol() -> None:
    assert isinstance(_FailingExtractor(), Extractor)

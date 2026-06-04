"""LangGraph orchestration: classify -> extract -> retrieve/answer -> recommend.

A deterministic, typed state machine (no LLM required), so it is fully testable
in CI. Conditional edges encode the routing logic:

    classify
      invoice -> extract -> answer? -> recommend -> END
      non-invoice -> recommend -> END

The same graph accepts an LLM-driven variant by swapping the static edges for a
tool-calling node over `tools.build_structured_tools(...)`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Literal

from langgraph.graph import END, StateGraph

from invoice_iq.agent import tools
from invoice_iq.agent.state import AgentResult, AgentState, NextAction
from invoice_iq.classifier.predict import DocumentTypePredictor
from invoice_iq.extraction.base import ExtractionError, Extractor
from invoice_iq.ingestion.base import OCRProvider
from invoice_iq.ingestion.local_ocr import LocalOCRProvider, ingest_pdf
from invoice_iq.rag.qa import RAGPipeline
from invoice_iq.schemas.documents import DocumentType


@dataclass
class AgentDeps:
    """Components the agent orchestrates (injected for testability)."""

    predictor: DocumentTypePredictor
    extractor: Extractor
    rag: RAGPipeline
    ocr_provider: OCRProvider = field(default_factory=LocalOCRProvider)
    min_confidence: float = tools.DEFAULT_MIN_CONFIDENCE
    review_total: Decimal = tools.DEFAULT_REVIEW_TOTAL


def build_agent(deps: AgentDeps) -> object:
    """Build and compile the LangGraph agent for the given dependencies."""

    def classify_node(state: AgentState) -> dict[str, object]:
        raw, ocr = ingest_pdf(Path(state["pdf_path"]), deps.ocr_provider)
        doc_type, confidence = tools.classify_document(deps.predictor, ocr)
        return {
            "raw_document": raw,
            "ocr": ocr,
            "doc_id": raw.doc_id,
            "document_type": doc_type,
            "classification_confidence": confidence,
            "trace": [f"classified as {doc_type.value} (confidence {confidence:.2f})"],
        }

    def extract_node(state: AgentState) -> dict[str, object]:
        ocr = state["ocr"]
        assert ocr is not None
        try:
            invoice = tools.extract_invoice(deps.extractor, ocr)
        except ExtractionError as exc:
            return {"errors": [str(exc)], "trace": ["extraction failed"]}
        return {"invoice": invoice, "trace": [f"extracted invoice {invoice.invoice_number}"]}

    def answer_node(state: AgentState) -> dict[str, object]:
        invoice = state["invoice"]
        question = state["question"]
        assert invoice is not None and question is not None
        answer = tools.answer_question(deps.rag, invoice, state["doc_id"], question)
        return {"answer": answer, "trace": [f"answered: {question}"]}

    def recommend_node(state: AgentState) -> dict[str, object]:
        action_str, rationale = tools.recommend_action(
            document_type=state["document_type"],
            classification_confidence=state["classification_confidence"],
            invoice=state.get("invoice"),
            errors=state.get("errors", []),
            min_confidence=deps.min_confidence,
            review_total=deps.review_total,
        )
        action = NextAction(action_str)
        return {
            "recommendation": action,
            "rationale": rationale,
            "trace": [f"recommend: {action.value}"],
        }

    def route_after_classify(state: AgentState) -> Literal["extract", "recommend"]:
        if state["document_type"] is DocumentType.INVOICE:
            return "extract"
        return "recommend"

    def route_after_extract(state: AgentState) -> Literal["answer", "recommend"]:
        if state.get("errors"):
            return "recommend"
        if state.get("question"):
            return "answer"
        return "recommend"

    graph: StateGraph = StateGraph(AgentState)
    graph.add_node("classify", classify_node)
    graph.add_node("extract", extract_node)
    graph.add_node("answer", answer_node)
    graph.add_node("recommend", recommend_node)

    graph.set_entry_point("classify")
    graph.add_conditional_edges(
        "classify", route_after_classify, {"extract": "extract", "recommend": "recommend"}
    )
    graph.add_conditional_edges(
        "extract", route_after_extract, {"answer": "answer", "recommend": "recommend"}
    )
    graph.add_edge("answer", "recommend")
    graph.add_edge("recommend", END)
    return graph.compile()


def run_agent(deps: AgentDeps, pdf_path: Path | str, question: str | None = None) -> AgentResult:
    """Run the agent end-to-end on one PDF and return the typed result."""
    app = build_agent(deps)
    initial: AgentState = {
        "pdf_path": str(pdf_path),
        "question": question,
        "errors": [],
        "trace": [],
    }
    final = app.invoke(initial)  # type: ignore[attr-defined]
    return AgentResult(
        document_type=final["document_type"],
        classification_confidence=final["classification_confidence"],
        invoice=final.get("invoice"),
        answer=final.get("answer"),
        recommendation=final["recommendation"],
        rationale=final["rationale"],
        errors=final.get("errors", []),
        trace=final.get("trace", []),
    )

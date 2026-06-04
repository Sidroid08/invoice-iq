"""Typed tools the agent orchestrates.

Each capability (classify / extract / answer / recommend) is a typed function with
a clear contract. The LangGraph nodes call these directly (deterministic, testable),
and `build_structured_tools` also exposes them as langchain-core `StructuredTool`s
with Pydantic argument schemas - the same tools an LLM-driven agent would call.
"""

from __future__ import annotations

from decimal import Decimal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict

from invoice_iq.classifier.predict import DocumentTypePredictor
from invoice_iq.extraction.base import Extractor
from invoice_iq.rag.qa import Answer, RAGPipeline
from invoice_iq.schemas.documents import DocumentType, OCRResult
from invoice_iq.schemas.invoice import Invoice

# Defaults for the recommendation policy.
DEFAULT_MIN_CONFIDENCE = 0.60
DEFAULT_REVIEW_TOTAL = Decimal("5000")


class _ClassifyDocumentArgs(BaseModel):
    """Arguments for the classify_document tool."""

    model_config = ConfigDict(extra="forbid")

    ocr: OCRResult


class _ExtractInvoiceArgs(BaseModel):
    """Arguments for the extract_invoice tool."""

    model_config = ConfigDict(extra="forbid")

    ocr: OCRResult


class _AnswerQuestionArgs(BaseModel):
    """Arguments for the answer_question tool."""

    model_config = ConfigDict(extra="forbid")

    invoice: Invoice
    doc_id: str
    question: str


def classify_document(
    predictor: DocumentTypePredictor, ocr: OCRResult
) -> tuple[DocumentType, float]:
    """Classify OCR'd text into a document type with a confidence score."""
    return predictor.predict_ocr(ocr)


def extract_invoice(extractor: Extractor, ocr: OCRResult) -> Invoice:
    """Extract a validated `Invoice` from OCR text (raises `ExtractionError`)."""
    return extractor.extract(ocr)


def answer_question(rag: RAGPipeline, invoice: Invoice, doc_id: str, question: str) -> Answer:
    """Index the invoice and answer a question about it via retrieval."""
    rag.index_invoice(invoice, doc_id)
    return rag.ask(question)


def recommend_action(
    document_type: DocumentType,
    classification_confidence: float,
    invoice: Invoice | None,
    errors: list[str],
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    review_total: Decimal = DEFAULT_REVIEW_TOTAL,
) -> tuple[str, str]:
    """Decide the next action from the accumulated state. Returns (action, rationale).

    Action is a `NextAction` value string (kept as str to avoid importing state here).
    """
    if document_type is not DocumentType.INVOICE:
        return (
            "route_non_invoice",
            f"Document classified as {document_type.value}; route to its workflow.",
        )
    if errors:
        return "review", f"Invoice extraction failed ({errors[0]}); needs human review."
    if classification_confidence < min_confidence:
        return (
            "review",
            f"Low classification confidence ({classification_confidence:.2f}); review recommended.",
        )
    if invoice is not None and invoice.total.amount > review_total:
        return (
            "review",
            f"High-value invoice ({invoice.total}); requires approval before payment.",
        )
    return "approve", "Invoice extracted and validated; approve for payment."


def build_structured_tools(
    predictor: DocumentTypePredictor, extractor: Extractor, rag: RAGPipeline
) -> list[StructuredTool]:
    """Expose the capabilities as langchain-core StructuredTools (for LLM-driven agents)."""

    def _classify_document(ocr: OCRResult) -> tuple[DocumentType, float]:
        return classify_document(predictor, ocr)

    def _extract_invoice(ocr: OCRResult) -> Invoice:
        return extract_invoice(extractor, ocr)

    def _answer_question(invoice: Invoice, doc_id: str, question: str) -> Answer:
        return answer_question(rag, invoice, doc_id, question)

    return [
        StructuredTool.from_function(
            func=_classify_document,
            name="classify_document",
            description="Classify a document (invoice/receipt/contract) from its OCR result.",
            args_schema=_ClassifyDocumentArgs,
        ),
        StructuredTool.from_function(
            func=_extract_invoice,
            name="extract_invoice",
            description="Extract structured, validated invoice fields from an OCR result.",
            args_schema=_ExtractInvoiceArgs,
        ),
        StructuredTool.from_function(
            func=_answer_question,
            name="answer_question",
            description="Answer a question about an invoice using retrieval-augmented generation.",
            args_schema=_AnswerQuestionArgs,
        ),
    ]

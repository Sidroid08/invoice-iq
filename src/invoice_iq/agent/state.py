"""Typed state and result schemas for the LangGraph agent."""

from __future__ import annotations

from enum import Enum
from operator import add
from typing import Annotated, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from invoice_iq.rag.qa import Answer
from invoice_iq.schemas.documents import DocumentType, OCRResult, RawDocument
from invoice_iq.schemas.invoice import Invoice


class NextAction(str, Enum):
    """The recommended next action emitted by the agent's final node."""

    APPROVE = "approve"
    REVIEW = "review"
    REQUEST_INFO = "request_info"
    ROUTE_NON_INVOICE = "route_non_invoice"


class AgentState(TypedDict, total=False):
    """Mutable graph state. List fields use `add` reducers so nodes can append.

    Nodes return partial dicts that LangGraph merges into this state.
    """

    # Inputs
    pdf_path: str
    question: str | None

    # Populated as the graph runs
    doc_id: str
    raw_document: RawDocument | None
    ocr: OCRResult | None
    document_type: DocumentType
    classification_confidence: float
    invoice: Invoice | None
    answer: Answer | None
    recommendation: NextAction
    rationale: str

    # Accumulated across nodes
    errors: Annotated[list[str], add]
    trace: Annotated[list[str], add]


class AgentResult(BaseModel):
    """The agent's final, typed output (suitable for API responses)."""

    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType
    classification_confidence: float
    invoice: Invoice | None = None
    answer: Answer | None = None
    recommendation: NextAction
    rationale: str
    errors: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)

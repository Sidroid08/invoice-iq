"""API request/response schemas for the FastAPI serving layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from invoice_iq.agent.state import AgentResult
from invoice_iq.rag.qa import Answer
from invoice_iq.schemas.documents import DocumentType, OCRResult, RawDocument
from invoice_iq.schemas.invoice import Invoice


class HealthResponse(BaseModel):
    """Basic service health plus provider wiring."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    version: str
    providers: dict[str, str]
    vector_count: int


class IngestResponse(BaseModel):
    """PDF ingestion response."""

    model_config = ConfigDict(extra="forbid")

    raw_document: RawDocument
    ocr: OCRResult


class ClassificationResponse(BaseModel):
    """Document classification response."""

    model_config = ConfigDict(extra="forbid")

    raw_document: RawDocument
    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)
    ocr_provider: str
    page_count: int = Field(ge=0)


class PredictionInstance(BaseModel):
    """A text-only prediction instance for Vertex custom-container serving."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)


class PredictionRequest(BaseModel):
    """Vertex-compatible custom-container predict request."""

    model_config = ConfigDict(extra="forbid")

    instances: list[PredictionInstance] = Field(min_length=1)


class Prediction(BaseModel):
    """One document-type prediction."""

    model_config = ConfigDict(extra="forbid")

    document_type: DocumentType
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Vertex-compatible custom-container predict response."""

    model_config = ConfigDict(extra="forbid")

    predictions: list[Prediction]


class ExtractionResponse(BaseModel):
    """Invoice extraction response."""

    model_config = ConfigDict(extra="forbid")

    raw_document: RawDocument
    invoice: Invoice
    extractor: str


class AskResponse(BaseModel):
    """RAG answer response over one uploaded invoice."""

    model_config = ConfigDict(extra="forbid")

    raw_document: RawDocument
    invoice: Invoice
    answer: Answer


class AgentRunResponse(AgentResult):
    """Named API response for the Phase 5 agent result."""

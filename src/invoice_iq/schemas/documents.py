"""Document-level schemas: the raw input and OCR output contracts.

These sit at the *front* of the pipeline (ingestion → OCR) and feed both the
classifier (document type) and the extractor (structured fields).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentType(str, Enum):
    """The document classes the PyTorch classifier distinguishes.

    `invoice` is the project's focus; `receipt` and `contract` are contrast
    classes that make the classifier a genuine multi-class problem. `unknown`
    is the safe default before/under low-confidence classification.
    """

    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    UNKNOWN = "unknown"


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class RawDocument(BaseModel):
    """A source document handed to the pipeline, before any OCR/parsing.

    We deliberately store metadata + a reference rather than raw bytes so the
    schema stays cheap to log, serialize, and pass between stages.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(
        default_factory=lambda: uuid4().hex, description="Stable id for the document."
    )
    filename: str = Field(min_length=1, description="Original file name, e.g. 'acme-2026-04.pdf'.")
    content_type: str = Field(default="application/pdf", description="MIME type of the source.")
    size_bytes: int = Field(ge=0, default=0, description="File size in bytes.")
    num_pages: int = Field(ge=0, default=0, description="Page count, 0 if unknown.")
    ingested_at: datetime = Field(default_factory=_utcnow)

    @field_validator("filename")
    @classmethod
    def _strip_filename(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("filename must not be blank")
        return v


class OCRResult(BaseModel):
    """Text + light layout extracted from a `RawDocument` by an OCRProvider.

    `pages` holds per-page text (index 0 = page 1); `full_text` is the joined
    convenience view used by the classifier and rule-based extractor.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(description="Links back to the RawDocument.doc_id.")
    full_text: str = Field(description="All extracted text, pages joined by form feeds.")
    pages: list[str] = Field(default_factory=list, description="Per-page extracted text.")
    provider: str = Field(
        default="local", description="OCRProvider that produced this, e.g. 'local'."
    )
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Provider confidence 0-1."
    )

    @field_validator("full_text")
    @classmethod
    def _require_text(cls, v: str) -> str:
        # Empty OCR is almost always an upstream failure; fail fast and loud.
        if not v.strip():
            raise ValueError("OCRResult.full_text is empty — OCR/parse likely failed")
        return v

    @property
    def page_count(self) -> int:
        return len(self.pages)
